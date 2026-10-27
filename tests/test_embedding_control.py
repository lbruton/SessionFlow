"""SESF-6 red-phase tests for local-only embedding resource controls.

Requirements: 6.6, 6.7, 7.
"""

import time


def test_embedding_budget_reads_local_resource_limits_from_env(monkeypatch):
    from embedding_control import EmbeddingBudget

    monkeypatch.setenv("SESSIONFLOW_EMBED_BATCH_SIZE", "8")
    monkeypatch.setenv("SESSIONFLOW_EMBED_COOLDOWN_MS", "250")
    monkeypatch.setenv("SESSIONFLOW_BACKFILL_MAX_TURNS_PER_RUN", "50")
    monkeypatch.setenv("SESSIONFLOW_BACKFILL_RECENT_DAYS", "14")
    monkeypatch.setenv("SESSIONFLOW_BACKFILL_MODE", "recent")

    budget = EmbeddingBudget.from_env()

    assert budget.batch_size == 8
    assert budget.cooldown_ms == 250
    assert budget.max_turns_per_run == 50
    assert budget.recent_days == 14
    assert budget.mode == "recent"


def test_embedding_budget_pause_blocks_backfill_batches(monkeypatch):
    from embedding_control import EmbeddingBudget

    monkeypatch.setenv("SESSIONFLOW_BACKFILL_PAUSED", "true")

    budget = EmbeddingBudget.from_env()

    assert budget.paused is True
    assert budget.before_batch(batch_size=1, estimated_chars=20).allowed is False
    assert "paused" in budget.before_batch(batch_size=1, estimated_chars=20).reason.lower()


def test_embedding_identity_is_local_mlx_and_ignores_openai_env(monkeypatch):
    from embedding_control import EmbeddingIdentity

    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-not-used")
    monkeypatch.setenv("SESSIONFLOW_MODEL", "embeddinggemma")

    identity = EmbeddingIdentity.current_local()

    assert identity.embedding_provider == "local_mlx"
    assert identity.model_name == "embeddinggemma"
    assert identity.dimension == 768
    assert not hasattr(identity, "api_key")


def test_default_max_turns_per_run_is_continuous_safe():
    """SESF-12: a fresh-from-env budget must not silently cap a long-running server."""
    from embedding_control import DEFAULT_MAX_TURNS_PER_RUN, EmbeddingBudget

    assert DEFAULT_MAX_TURNS_PER_RUN >= 10_000
    assert EmbeddingBudget().max_turns_per_run == DEFAULT_MAX_TURNS_PER_RUN
    assert EmbeddingBudget.from_env().max_turns_per_run == DEFAULT_MAX_TURNS_PER_RUN


def test_budget_warns_first_time_cap_is_hit(caplog):
    """SESF-12: operator must see a distinct WARN the first time the cap denies inserts."""
    import logging

    from embedding_control import EmbeddingBudget

    budget = EmbeddingBudget(batch_size=4, cooldown_ms=0, max_turns_per_run=4)
    budget.after_batch(duration=0.01, turns=4)

    with caplog.at_level(logging.WARNING, logger="sessionflow.embedding_budget"):
        first = budget.before_batch(batch_size=4)
        second = budget.before_batch(batch_size=4)

    assert first.allowed is False
    assert second.allowed is False
    warns = [r for r in caplog.records if "MAX_TURNS_PER_RUN" in r.getMessage()]
    assert len(warns) == 1, "expected exactly one WARN across repeated denials"


def test_embedding_budget_records_cooldown_after_batch():
    from embedding_control import EmbeddingBudget

    budget = EmbeddingBudget(batch_size=4, cooldown_ms=200, max_turns_per_run=10)
    budget.after_batch(duration=0.05, turns=4)
    decision = budget.before_batch(batch_size=4, estimated_chars=100)

    assert decision.allowed is False
    assert decision.retry_after_seconds > 0
    time.sleep(decision.retry_after_seconds)
    assert budget.before_batch(batch_size=4, estimated_chars=100).allowed is True


# ---------------------------------------------------------------------------
# Identity per collection, and the waiting throttle memory embeds share with
# transcripts. The throttle never denies work; it waits out the cooldown.
# ---------------------------------------------------------------------------


def test_identity_records_the_collection_it_describes():
    from embedding_control import EmbeddingIdentity

    assert EmbeddingIdentity.current_local().collection_name == "sessions"
    assert EmbeddingIdentity.current_local(collection="memories").collection_name == "memories"


def test_throttle_keeps_the_200ms_floor():
    from embedding_control import EmbedThrottle

    assert EmbedThrottle(cooldown_ms=50).cooldown_ms == 200
    assert EmbedThrottle(cooldown_ms=350).cooldown_ms == 350


def test_throttle_waits_instead_of_dropping():
    import asyncio

    from embedding_control import EmbedThrottle

    clock = [100.0]
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    throttle = EmbedThrottle(cooldown_ms=200, clock=lambda: clock[0], sleep=fake_sleep)

    async def two_batches():
        await throttle.wait()
        throttle.done()
        clock[0] += 0.05
        await throttle.wait()
        throttle.done()

    asyncio.run(two_batches())

    assert slept == [_approx(0.15)]


def test_throttle_does_not_wait_after_the_cooldown():
    import asyncio

    from embedding_control import EmbedThrottle

    clock = [0.0]
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    throttle = EmbedThrottle(cooldown_ms=200, clock=lambda: clock[0], sleep=fake_sleep)

    async def batches():
        await throttle.wait()
        throttle.done()
        clock[0] += 1.0
        await throttle.wait()

    asyncio.run(batches())
    assert slept == []


def test_shared_throttle_is_one_instance():
    from embedding_control import get_embed_throttle

    assert get_embed_throttle() is get_embed_throttle()


def _approx(value):
    import pytest

    return pytest.approx(value, abs=1e-9)


def test_throttle_batch_spaces_concurrent_batches_by_the_cooldown():
    import asyncio

    from embedding_control import EmbedThrottle

    clock = [100.0]

    async def fake_sleep(seconds):
        clock[0] += seconds

    throttle = EmbedThrottle(cooldown_ms=200, clock=lambda: clock[0], sleep=fake_sleep)
    starts = []

    async def one_batch():
        async with throttle.batch():
            starts.append(clock[0])
            await asyncio.sleep(0)

    async def run():
        await asyncio.gather(one_batch(), one_batch())

    asyncio.run(run())
    asyncio.run(run())  # a second event loop must not hit a lock bound to the first

    assert starts[1] - starts[0] >= 0.2 - 1e-9
    assert starts[3] - starts[2] >= 0.2 - 1e-9


def test_concurrent_memory_embeds_are_spaced_by_the_cooldown(monkeypatch):
    import asyncio

    import embedding_control
    import memory_index
    import rag_engine

    clock = [50.0]

    async def fake_sleep(seconds):
        clock[0] += seconds

    throttle = embedding_control.EmbedThrottle(
        cooldown_ms=200, clock=lambda: clock[0], sleep=fake_sleep)
    monkeypatch.setattr(embedding_control, "get_embed_throttle", lambda: throttle)
    monkeypatch.setattr(rag_engine, "_embed_executor", None)
    starts = []

    def fake_embed(texts):
        starts.append(clock[0])
        time.sleep(0.05)  # a batch still running when the other caller reaches the throttle
        return [[0.0] for _ in texts]

    monkeypatch.setattr(rag_engine, "embed_texts", fake_embed)

    async def both():
        await asyncio.gather(memory_index.embed_for_index(["a"]),
                             memory_index.embed_for_index(["b"]))

    asyncio.run(both())

    assert len(starts) == 2
    assert starts[1] - starts[0] >= 0.2 - 1e-9


def test_throttle_reports_the_cooldown_left():
    from embedding_control import EmbedThrottle

    clock = [10.0]
    throttle = EmbedThrottle(cooldown_ms=200, clock=lambda: clock[0])

    assert throttle.remaining() == 0.0
    throttle.done()
    clock[0] += 0.05
    assert throttle.remaining() == _approx(0.15)
    clock[0] += 1.0
    assert throttle.remaining() == 0.0


def test_transcript_batch_waits_out_a_memory_embeds_cooldown(monkeypatch):
    import asyncio

    import embedding_control
    import memory_index
    import rag_engine

    throttle = embedding_control.EmbedThrottle(cooldown_ms=200)
    monkeypatch.setattr(embedding_control, "get_embed_throttle", lambda: throttle)
    monkeypatch.setattr(rag_engine, "_embed_executor", None)
    monkeypatch.setattr(rag_engine, "embed_texts", lambda texts: [[0.0] for _ in texts])
    asyncio.run(memory_index.embed_for_index(["memory"]))

    budget = embedding_control.EmbeddingBudget(batch_size=4, cooldown_ms=200, max_turns_per_run=10)
    decision = budget.before_batch(batch_size=4)

    assert decision.allowed is False
    assert 0 < decision.retry_after_seconds <= 0.2


def test_memory_embed_queued_behind_a_transcript_batch_waits_on_the_mlx_thread(monkeypatch):
    import asyncio

    import embedding_control
    import memory_index
    import rag_engine

    clock = [100.0]
    slept = []

    def fake_blocking_sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    throttle = embedding_control.EmbedThrottle(
        cooldown_ms=200, clock=lambda: clock[0], blocking_sleep=fake_blocking_sleep)
    monkeypatch.setattr(embedding_control, "get_embed_throttle", lambda: throttle)
    monkeypatch.setattr(rag_engine, "_embed_executor", None)
    starts = []

    def fake_embed(texts):
        starts.append(clock[0])
        return [[0.0] for _ in texts]

    monkeypatch.setattr(rag_engine, "embed_texts", fake_embed)
    throttle.done()  # a transcript batch finished after the memory embed was queued

    asyncio.run(memory_index._embed_on_mlx_thread(["memory"]))

    assert slept == [_approx(0.2)]
    assert starts == [_approx(100.2)]
    assert throttle.remaining() == _approx(0.2)


def test_throttle_batch_can_leave_the_stamp_to_the_caller():
    import asyncio

    from embedding_control import EmbedThrottle

    clock = [5.0]
    throttle = EmbedThrottle(cooldown_ms=200, clock=lambda: clock[0])

    async def run():
        async with throttle.batch(stamp=False):
            pass

    asyncio.run(run())
    assert throttle.remaining() == 0.0


def test_memory_embed_stamps_the_throttle_once_on_the_mlx_thread(monkeypatch):
    # A second stamp from the event loop would move the deadline under a
    # transcript batch that is already sleeping out the first one.
    import asyncio
    import threading

    import embedding_control
    import memory_index
    import rag_engine

    stamps = []

    class _Counting(embedding_control.EmbedThrottle):
        def done(self):
            stamps.append(threading.current_thread().name)
            super().done()

    throttle = _Counting(cooldown_ms=200)
    monkeypatch.setattr(embedding_control, "get_embed_throttle", lambda: throttle)
    monkeypatch.setattr(rag_engine, "_embed_executor", None)
    monkeypatch.setattr(rag_engine, "embed_texts", lambda texts: [[0.0] for _ in texts])

    asyncio.run(memory_index.embed_for_index(["memory"]))

    assert len(stamps) == 1
    assert stamps[0] != threading.main_thread().name
