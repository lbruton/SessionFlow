"""TDD red-phase tests for the SESF-41 secret-redaction guard.

Two cohorts live here:

* **B.1** — the pure ``secret_redaction.redact`` engine (detector + masker).
* **B.2** — the ``rag_engine.add_turns`` hook, config, report mode, and AC-17 scrub
  (appended below B.1).

Every token in this file is **synthetic and non-functional** — no raw secret value
appears anywhere (AC-18). The provider-shaped fixtures are crafted to match the real
``detect-secrets`` plugin regexes (verified against ``detect-secrets 1.5.0``) and the
custom Tier-1 regexes ``secret_redaction.py`` adds for Anthropic / Google / HuggingFace,
so the Cohort-C implementation can make them green without weakening any assertion.

RED gate: against the A.2 stub (``redact`` returns ``(text, [])``) every acceptance
criterion below has at least one assertion that fails; each negative/static guardrail
is paired with a positive control that fails on the stub.
"""

from __future__ import annotations

import importlib

import pytest

secret_redaction = importlib.import_module("secret_redaction")
redact = secret_redaction.redact


# --- Synthetic, non-functional fixtures (AC-18) -----------------------------

# Every provider-shaped token is ASSEMBLED FROM PARTS so no contiguous secret
# literal appears in this source file. This defeats GitHub push-protection /
# secret scanning (which scans the raw blob) while the runtime value is the full
# token the detector must flag. All values are synthetic and non-functional.
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
AWS_KEY2 = "AKIA" + "1234567890ABCDEF"  # second AWS key for the determinism test
GH_KEY = "ghp" + "_" + "0123456789abcdefABCDEF0123456789abcd"
GITLAB_KEY = "glpat" + "-ABCDEFGHIJ1234567890"
SLACK_KEY = "xoxb" + "-0000000000-0000000000-ABCDEFGHIJKLMNOPQRSTUVWX"
STRIPE_KEY = "sk_live" + "_0123456789abcdefABCD1234"
OPENAI_KEY = "sk-" + "A" * 20 + "T3Blbk" + "FJ" + "B" * 20
ANTHROPIC_KEY = "sk-ant-" + "api03-" + "A" * 40
GOOGLE_KEY = "AIza" + "0123456789abcdefghij0123456789ABCDE"
HF_KEY = "hf" + "_0123456789abcdefABCDEF0123456789ab"
NPM_TOKEN = "npm" + "_0123456789abcdefABCDEF0123456789abcd"

# Tier-1 structured provider tokens → expected canonical rule name. Shapes match
# detect-secrets 1.5.0 built-ins or the custom Tier-1 regexes.
TIER1 = {
    "AWS": (AWS_KEY, "AWS"),
    "GITHUB": (GH_KEY, "GITHUB"),
    "GITLAB": (GITLAB_KEY, "GITLAB"),
    "SLACK": (SLACK_KEY, "SLACK"),
    "STRIPE": (STRIPE_KEY, "STRIPE"),
    "OPENAI": (OPENAI_KEY, "OPENAI"),
    "ANTHROPIC": (ANTHROPIC_KEY, "ANTHROPIC"),
    "GOOGLE": (GOOGLE_KEY, "GOOGLE_API_KEY"),
    "HUGGINGFACE": (HF_KEY, "HUGGINGFACE"),
}

# npm needs .npmrc assignment context to match NpmDetector (verified).
NPM_LINE = "//registry.npmjs.org/:_authToken=" + NPM_TOKEN

# JWT — three base64url segments (header.payload.signature), all synthetic.
JWT_TOKEN = "eyJ" + "hbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0." + "A" * 39

# PEM block — a synthetic, non-functional private key. D-6 masks the whole block.
PEM_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIBOwIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n"
    "KUpRKfFLfRYC9AIKjbJTWit+CqvjWYzvQwECAwEAAQJAfaketestkeybodyXXXXXX\n"
    "-----END RSA PRIVATE KEY-----"
)

# A high-entropy token matching NO provider format (Tier-3 entropy only).
ENTROPY_TOKEN = "Zk9wYmFyQmF6UXV4OTk4MjM3NDYxMjkwTm9wZQ"

# A 32-char high-entropy token (entropy >= 4.0, not UUID / 40hex / 64hex) used by
# the SESF-42 snippet-leak regression. Assembled from parts so no contiguous secret
# literal appears in this source. It is forced (keyword-adjacent within 40 chars) in
# the full line but sits OUTSIDE the +-24 snippet window of its forcing keyword.
ENTROPY_FORCED_TOKEN = "Xk7Qm2Wp9Lr4Bn8" + "Zt5Cv3Hd6Fj1Gs0Yo"

# Allowlist shapes that MUST NOT be flagged (AC-9).
UUID_VALUE = "550e8400-e29b-41d4-a716-446655440000"
GIT_SHA40 = "5e150f177902427c6f79e1713b608de8f4c5964a"
SHA256_HEX = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# content-hash doc_id == "{provider}:{session}:{sha256-hexdigest}" (the hex tail
# is the content hash; covered by the same hex-shape allowlist).
DOC_ID_HASH = SHA256_HEX
ISSUE_ID = "SESF-41"

# Tier-2 contextual assignment (keyword + ':'/'=' + value), incl. the MCP-config
# dump leak class. Value is non-provider-shaped so only Tier-2 catches it.
ASSIGN_SECRET_VALUE = "aB3xY7zQ9mK2pL5nR8tW"
MCP_CONFIG_DUMP = (
    '{"mcpServers": {"svc": {"env": {"MY_SERVICE_TOKEN": "%s"}}}}'
    % ASSIGN_SECRET_VALUE
)


def _rules(hits):
    """Return the set of rule names across a hit list."""
    return {h.rule_name for h in hits}


# === B.1 — pure redact engine ===============================================


@pytest.mark.parametrize("key", sorted(TIER1))
def test_ac5_tier1_provider_token_masked_with_rule_name(key):
    """AC-5: each Tier-1 provider class masks with its rule name (tier 1)."""
    value, rule = TIER1[key]
    text = f"my credential is {value} ok"
    out, hits = redact(text, mode="enforce")
    # Positive control (fails on stub): the value is gone, placeholder present.
    assert value not in out
    assert f"[REDACTED:{rule}]" in out
    assert rule in _rules(hits)
    assert any(h.tier == 1 for h in hits if h.rule_name == rule)


def test_ac5_npm_token_in_npmrc_context():
    """AC-5: npm token in .npmrc assignment context is masked as NPM (tier 1)."""
    out, hits = redact(NPM_LINE, mode="enforce")
    assert NPM_TOKEN not in out
    assert "[REDACTED:NPM]" in out
    assert "NPM" in _rules(hits)


def test_ac5_jwt_masked():
    """AC-5: a JWT is masked as JWT (tier 1)."""
    out, hits = redact(f"bearer {JWT_TOKEN}", mode="enforce")
    assert JWT_TOKEN not in out
    assert "[REDACTED:JWT]" in out
    assert "JWT" in _rules(hits)


def test_ac5_pem_private_key_block_masked_whole(tmp_path):
    """AC-5/D-6: the whole PEM block is masked as PRIVATE_KEY, not just the marker."""
    out, hits = redact(f"key:\n{PEM_BLOCK}\nend", mode="enforce")
    # The key body line must not survive (D-6 multi-line pre-pass).
    assert "MIIBOwIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu" not in out
    assert "[REDACTED:PRIVATE_KEY]" in out
    assert "PRIVATE_KEY" in _rules(hits)


def test_ac4_placeholder_format_and_fallback():
    """AC-4: typed placeholder format, with [REDACTED:SECRET] as the no-name fallback."""
    # Named format (positive control — fails on the stub).
    out, _ = redact(f"cred {AWS_KEY} done", mode="enforce")
    assert "[REDACTED:AWS]" in out
    # Fallback rule name when a detector has no specific name.
    assert secret_redaction._placeholder("") == "[REDACTED:SECRET]"
    assert secret_redaction._placeholder("AWS") == "[REDACTED:AWS]"


def test_ac6_tier2_assignment_masked():
    """AC-6: a contextual keyword=value assignment is masked (value only, tier 2)."""
    text = f"api_key: {ASSIGN_SECRET_VALUE}"
    out, hits = redact(text, mode="enforce")
    assert ASSIGN_SECRET_VALUE not in out
    assert "api_key:" in out  # the key is preserved; only the value is masked
    assert "[REDACTED:" in out
    assert any(h.tier == 2 for h in hits)


def test_ac6_mcp_config_dump_leak_class():
    """AC-6: the MCP-config-dump leak class (JSON env value) is caught by Tier-2."""
    out, hits = redact(MCP_CONFIG_DUMP, mode="enforce")
    assert ASSIGN_SECRET_VALUE not in out
    assert "[REDACTED:" in out
    assert any(h.tier == 2 for h in hits)


@pytest.mark.parametrize("placeholder", ["***", "<your-key>", "changeme", "xxxx", "ab"])
def test_ac7_placeholder_shape_or_short_value_not_flagged(placeholder):
    """AC-7: placeholder-shape or sub-minimum-length values are NOT flagged.

    Paired with a positive control so the test still fails on the stub.
    """
    out, hits = redact(f"api_key: {placeholder}", mode="enforce")
    assert placeholder in out  # not redacted
    assert not any(h.tier == 2 for h in hits)
    # Positive control (fails on stub): a real value IS flagged.
    out2, hits2 = redact(f"api_key: {ASSIGN_SECRET_VALUE}", mode="enforce")
    assert ASSIGN_SECRET_VALUE not in out2


def test_ac8_isolated_entropy_reported_not_masked():
    """AC-8: an isolated high-entropy token is advisory — reported, not masked."""
    text = f"random blob {ENTROPY_TOKEN} trailing"
    out, hits = redact(text, mode="enforce")
    # Advisory: the entropy hit is reported...
    assert any(h.tier == 3 for h in hits)
    # ...but NOT auto-masked (no keyword adjacency, not in a fence).
    assert ENTROPY_TOKEN in out


def test_ac8_entropy_in_fenced_block_masked():
    """AC-8: a high-entropy token inside a ```-fenced block IS masked."""
    text = f"```\n{ENTROPY_TOKEN}\n```"
    out, hits = redact(text, mode="enforce")
    assert ENTROPY_TOKEN not in out
    assert "[REDACTED:" in out


@pytest.mark.parametrize(
    "allowed",
    [UUID_VALUE, GIT_SHA40, SHA256_HEX, DOC_ID_HASH, ISSUE_ID],
)
def test_ac9_structural_allowlist_not_flagged(allowed):
    """AC-9: UUID / git SHA / SHA-256 / doc_id hash / issue ID are not flagged."""
    out, hits = redact(f"reference {allowed} here", mode="enforce")
    assert allowed in out  # survives unredacted
    assert not hits
    # Positive control (fails on stub): a flaggable token alongside IS masked.
    out2, _ = redact(f"reference {allowed} and {AWS_KEY}", mode="enforce")
    assert AWS_KEY not in out2
    assert allowed in out2


def test_ac9_operator_supplied_allowlist_pattern():
    """AC-9: an operator-supplied pattern (via the allowlist arg) is not flagged."""
    import re

    token = AWS_KEY
    out, hits = redact(
        f"internal {token} ok",
        mode="enforce",
        allowlist=[re.compile(re.escape(token))],
    )
    assert token in out
    assert not hits
    # Positive control: without the allowlist the same token IS masked.
    out2, _ = redact(f"internal {token} ok", mode="enforce")
    assert token not in out2


def test_ac14_determinism_identical_bytes():
    """AC-14: redact over identical bytes yields byte-identical output."""
    text = f"key {TIER1['AWS'][0]} done"
    assert redact(text, mode="enforce") == redact(text, mode="enforce")


def test_ac14_determinism_differ_only_in_secret():
    """AC-14: two inputs differing only in secret values redact identically."""
    a = f"aws key {AWS_KEY} done"
    b = f"aws key {AWS_KEY2} done"
    assert redact(a, mode="enforce")[0] == redact(b, mode="enforce")[0]
    assert redact(a, mode="enforce")[0] == "aws key [REDACTED:AWS] done"


def test_ac15_idempotent_redact_of_redacted_text():
    """AC-15: redact(redact(x)) == redact(x); placeholders are left untouched."""
    x = f"token {TIER1['OPENAI'][0]} here"
    once = redact(x, mode="enforce")[0]
    # Positive control (fails on stub): the first pass actually changes the text.
    assert once != x
    twice = redact(once, mode="enforce")[0]
    assert twice == once


def test_ac15_existing_placeholder_not_rematched():
    """AC-15: an existing [REDACTED:...] placeholder is never re-wrapped."""
    text = "see [REDACTED:AWS] and [REDACTED:OPENAI] ok"
    out, _ = redact(text, mode="enforce")
    assert out == text


def test_ac16_no_io_during_redact(monkeypatch):
    """AC-16: redact performs no filesystem I/O (and still redacts correctly)."""
    import builtins
    import socket

    def _no_open(*args, **kwargs):
        raise AssertionError("redact must not open files")

    def _no_socket(*args, **kwargs):
        raise AssertionError("redact must not open sockets")

    monkeypatch.setattr(builtins, "open", _no_open)
    monkeypatch.setattr(socket, "socket", _no_socket)
    # Positive control (fails on stub): redaction still happens with no I/O.
    out, hits = redact(f"key {TIER1['AWS'][0]} done", mode="enforce")
    assert "[REDACTED:AWS]" in out
    assert hits


def test_ac18_hits_carry_no_raw_value():
    """AC-18: detected hits expose rule/tier only — never the raw secret value."""
    value = TIER1["AWS"][0]
    _, hits = redact(f"key {value} done", mode="report")
    # Positive control (fails on stub): the secret IS detected...
    assert hits
    # ...and no hit's metadata echoes the raw value (rule names / tiers only).
    for h in hits:
        assert value not in str(h)
        assert value not in h.rule_name
    assert "value" not in type(hits[0])._fields


# === B.2 — add_turns hook, config, report mode, AC-17 scrub =================

# Env var names the C.2 hook reads (contract pinned here).
ENV_ENABLE = "SESSIONFLOW_REDACT"
ENV_MODE = "SESSIONFLOW_REDACT_MODE"
ENV_ALLOWLIST = "SESSIONFLOW_REDACT_ALLOWLIST"

DB = "/tmp/sessionflow-redaction-test.db"  # mocked — never touched on disk
# AWS_KEY / GH_KEY are the part-assembled fixtures defined at the top of the file.


@pytest.fixture
def add_turns_harness(monkeypatch):
    """Wire ``add_turns`` with embed/Milvus/FTS mocked and env cleared.

    The Milvus client is a ``@contextmanager`` (rag_engine.py:668 yields), so the
    mock yields its fake client through ``contextlib.contextmanager`` to support
    ``with milvus_client(...) as client:`` (QWEN review concern #4).
    """
    import contextlib
    import types as _types

    import rag_engine

    for var in (ENV_ENABLE, ENV_MODE, ENV_ALLOWLIST):
        monkeypatch.delenv(var, raising=False)

    cap = {"embed_inputs": [], "milvus_data": None, "fts_records": None}

    def fake_embed_texts(texts, is_query=False):
        cap["embed_inputs"].append(list(texts))
        return [[0.0] * rag_engine._EMBED_DIM for _ in texts]

    monkeypatch.setattr(rag_engine, "embed_texts", fake_embed_texts)

    class _Decision:
        allowed = True
        retry_after_seconds = 0
        reason = ""

    class _Budget:
        def split_batches(self, turns):
            return [turns] if turns else []

        def before_batch(self, **kw):
            return _Decision()

        def after_batch(self, *a, **k):
            pass

    monkeypatch.setattr(rag_engine, "get_embedding_budget", lambda: _Budget())

    class _Client:
        def query(self, **kw):
            return []

        def insert(self, collection_name, data):
            cap["milvus_data"] = data

        def has_collection(self, name):
            return True

    @contextlib.contextmanager
    def _fake_milvus_client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _fake_milvus_client)

    def _fts_insert(conn, records):
        cap["fts_records"] = records

    fake_fts = _types.SimpleNamespace(
        connection=lambda db_path: object(),
        insert=_fts_insert,
        close_ephemeral=lambda conn: None,
    )
    monkeypatch.setattr(rag_engine, "_fts", fake_fts)

    # get_stats internals so AC-10/AC-11 can read the redaction surface.
    monkeypatch.setattr(rag_engine, "_query_all", lambda *a, **k: [])
    monkeypatch.setattr(
        rag_engine,
        "fts_lag_status",
        lambda **k: {"fts_row_count": 0, "fts_lag": 0, "fts_backfill_required": False},
    )

    # Reset durable per-rule counters between tests (once C.2 introduces them).
    if hasattr(rag_engine, "_redaction_counters"):
        rag_engine._redaction_counters.clear()

    return rag_engine, cap


def test_ac1_enforce_redacts_all_three_sinks(add_turns_harness, monkeypatch):
    """AC-1: enforce mode redacts embed input, Milvus document, and FTS content."""
    rag_engine, cap = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")
    rag_engine.add_turns([{"doc_id": "d1", "text": f"cred {AWS_KEY} end"}], db_path=DB)
    assert AWS_KEY not in cap["embed_inputs"][0][0]
    assert "[REDACTED:AWS]" in cap["embed_inputs"][0][0]
    assert AWS_KEY not in cap["milvus_data"][0]["document"]
    assert "[REDACTED:AWS]" in cap["milvus_data"][0]["document"]
    assert AWS_KEY not in cap["fts_records"][0]["content"]
    assert "[REDACTED:AWS]" in cap["fts_records"][0]["content"]


def test_ac2_embedding_over_redacted_text_identical(add_turns_harness, monkeypatch):
    """AC-2: two Turns differing only in secret values feed identical embed input."""
    rag_engine, cap = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")
    rag_engine.add_turns(
        [
            {"doc_id": "a", "text": f"key {AWS_KEY} done"},
            {"doc_id": "b", "text": f"key {AWS_KEY2} done"},
        ],
        db_path=DB,
    )
    batch = cap["embed_inputs"][0]
    assert batch[0] == batch[1] == "key [REDACTED:AWS] done"


def test_ac3_async_entry_point_also_redacted(add_turns_harness, monkeypatch):
    """AC-3: the single hook covers add_turns_async (no per-entry-point change)."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    rag_engine, cap = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(rag_engine, "_embed_executor", executor)

    async def _run():
        # Build sync primitives inside the running loop so they bind to it.
        monkeypatch.setattr(rag_engine, "_embed_semaphore", asyncio.Semaphore(1))
        monkeypatch.setattr(rag_engine, "_write_lock", asyncio.Lock())
        return await rag_engine.add_turns_async(
            [{"doc_id": "z", "text": f"cred {AWS_KEY} end"}], db_path=DB
        )

    try:
        asyncio.run(_run())
    finally:
        executor.shutdown(wait=True)  # don't leak the worker thread
    assert AWS_KEY not in cap["milvus_data"][0]["document"]
    assert "[REDACTED:AWS]" in cap["milvus_data"][0]["document"]


def test_ac3_redaction_runs_before_issue_extraction(add_turns_harness, monkeypatch):
    """AC-3: redaction runs before _extract_issue_ids — issue IDs survive."""
    rag_engine, cap = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")
    rag_engine.add_turns(
        [{"doc_id": "d", "text": f"SESF-41 uses {AWS_KEY} now"}], db_path=DB
    )
    data = cap["milvus_data"][0]
    assert AWS_KEY not in data["document"]
    assert "SESF-41" in data["document"]  # issue id is not a secret — survives
    assert ",SESF-41," in data["issue_ids"]  # extracted from the redacted text


def test_ac10_report_stores_raw_but_emits_counts(add_turns_harness, monkeypatch, caplog):
    """AC-10: report mode stores unredacted text but emits per-rule counts."""
    rag_engine, cap = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "report")
    with caplog.at_level("INFO"):
        rag_engine.add_turns([{"doc_id": "d", "text": f"cred {AWS_KEY} end"}], db_path=DB)
    # Stored UNREDACTED in every sink.
    assert AWS_KEY in cap["milvus_data"][0]["document"]
    assert AWS_KEY in cap["embed_inputs"][0][0]
    # Per-rule counters surfaced via get_stats (rule names only, no values).
    stats = rag_engine.get_stats(db_path=DB)
    assert stats["redaction"]["counts"].get("AWS", 0) >= 1
    # INFO histogram line names the rule but never the value.
    assert any(
        "AWS" in r.getMessage() and AWS_KEY not in r.getMessage()
        for r in caplog.records
    )


def test_ac11_unset_defaults_to_report_enabled(add_turns_harness):
    """AC-11: SESSIONFLOW_REDACT unset ⇒ enabled in report mode (the default)."""
    rag_engine, cap = add_turns_harness  # env vars cleared by the harness
    rag_engine.add_turns([{"doc_id": "d", "text": f"cred {AWS_KEY} end"}], db_path=DB)
    # Default report → unredacted storage...
    assert AWS_KEY in cap["milvus_data"][0]["document"]
    # ...but detection still ran (proves enabled-by-default).
    stats = rag_engine.get_stats(db_path=DB)
    assert stats["redaction"]["enabled"] is True
    assert stats["redaction"]["mode"] == "report"
    assert stats["redaction"]["counts"].get("AWS", 0) >= 1


def test_ac12_off_no_detection_no_mutation(add_turns_harness, monkeypatch):
    """AC-12: SESSIONFLOW_REDACT off ⇒ Turn text stored exactly as before."""
    rag_engine, cap = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "off")
    text = f"cred {AWS_KEY} end"
    rag_engine.add_turns([{"doc_id": "d", "text": text}], db_path=DB)
    assert cap["milvus_data"][0]["document"] == text
    assert cap["embed_inputs"][0][0] == text
    # Positive control (fails on the no-hook state): enforce DOES mutate.
    cap["milvus_data"] = None
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")
    rag_engine.add_turns([{"doc_id": "d2", "text": text}], db_path=DB)
    assert AWS_KEY not in cap["milvus_data"][0]["document"]


def test_ac13_mode_and_allowlist_path_read_from_env(add_turns_harness, monkeypatch, tmp_path):
    """AC-13: mode from SESSIONFLOW_REDACT_MODE; allowlist from the configured path."""
    rag_engine, cap = add_turns_harness
    allow = tmp_path / "allowlist.txt"
    allow.write_text(AWS_KEY + "\n")  # operator exempts this exact token
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")
    monkeypatch.setenv(ENV_ALLOWLIST, str(allow))
    rag_engine.add_turns([{"doc_id": "d", "text": f"{AWS_KEY} and {GH_KEY}"}], db_path=DB)
    doc = cap["milvus_data"][0]["document"]
    assert AWS_KEY in doc  # operator-allowlisted via the env-configured file → survives
    assert GH_KEY not in doc  # not allowlisted → masked (proves enforce mode from env)
    assert "[REDACTED:GITHUB]" in doc


def test_ac17_fts_insert_error_is_scrubbed(add_turns_harness, monkeypatch, caplog):
    """AC-17: a secret in the FTS-insert failure message is scrubbed from the log."""
    rag_engine, cap = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")

    def boom(conn, records):
        raise RuntimeError(f"insert failed near {AWS_KEY}")

    monkeypatch.setattr(rag_engine._fts, "insert", boom)
    with caplog.at_level("WARNING"):
        rag_engine.add_turns([{"doc_id": "d", "text": "benign text"}], db_path=DB)
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert AWS_KEY not in logged  # FTS failure is non-fatal AND scrubbed


@pytest.mark.parametrize(
    "invoke",
    [
        lambda rag: rag.delete_by_session("sess-1", db_path=DB),
        lambda rag: rag.delete_by_branch("feature/x", db_path=DB),
        lambda rag: rag.delete_older_than(30, db_path=DB),
    ],
    ids=["by_session", "by_branch", "older_than"],
)
def test_ac17_fts_delete_error_is_scrubbed(add_turns_harness, monkeypatch, caplog, invoke):
    """AC-17: a secret in any FTS delete failure message is scrubbed from the log.

    Covers the three cleanup entry points (``delete_by_session`` / ``delete_by_branch``
    use ``_fts.delete``; ``delete_older_than`` uses ``_fts.delete_where``). A SQLite
    error string can echo bound parameters or row content, so the non-fatal handler
    must route ``e`` through ``_scrub_exception`` rather than logging it raw.
    """
    rag_engine, _ = add_turns_harness

    def boom(*args, **kwargs):
        raise RuntimeError(f"delete failed near {AWS_KEY}")

    monkeypatch.setattr(rag_engine._fts, "delete", boom, raising=False)
    monkeypatch.setattr(rag_engine._fts, "delete_where", boom, raising=False)
    with caplog.at_level("WARNING"):
        invoke(rag_engine)  # non-fatal: the forced FTS error must not propagate
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert AWS_KEY not in logged  # delete failure is non-fatal AND scrubbed
    assert "[REDACTED:AWS]" in logged  # positive control: the scrub actually fired


def test_ac17_embedding_error_is_scrubbed(add_turns_harness, monkeypatch):
    """AC-17: a secret in a raised embedding error is scrubbed before re-raise."""
    rag_engine, _ = add_turns_harness
    monkeypatch.setenv(ENV_ENABLE, "on")
    monkeypatch.setenv(ENV_MODE, "enforce")

    def boom(texts, is_query=False):
        raise RuntimeError(f"embed failed near {AWS_KEY}")

    monkeypatch.setattr(rag_engine, "embed_texts", boom)
    with pytest.raises(Exception) as ei:
        rag_engine.add_turns([{"doc_id": "d", "text": "benign text"}], db_path=DB)
    assert AWS_KEY not in str(ei.value)


# --- Regression tests for PR #37 review findings ----------------------------


def test_ac9_tier2_hex_secret_is_not_structurally_allowlisted():
    """Regression: the 40/64-hex shape exemption is Tier-3 only.

    A hex-shaped secret in a Tier-2 assignment (`api_key=<64hex>`) must still be
    redacted — the structural allowlist must not blanket-exempt all-hex values.
    """
    hex64 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    out, _ = redact(f"api_key: {hex64}", mode="enforce")
    assert hex64 not in out  # Tier-2 hit — not suppressed by the hex shape
    assert "[REDACTED:" in out
    # The same hex as a bare token (Tier-3 entropy) IS structurally allowlisted.
    out2, _ = redact(f"commit {hex64} landed", mode="enforce")
    assert hex64 in out2


def test_ac8_keyword_adjacency_checks_all_occurrences():
    """Regression: entropy adjacency must consider every occurrence of the token.

    When a high-entropy token appears twice and only the later occurrence is
    keyword-adjacent, it must still be force-masked.
    """
    token = ENTROPY_TOKEN
    text = f"{token} appears, then api_key {token}"
    out, _ = redact(text, mode="enforce")
    assert token not in out  # the keyword-adjacent occurrence forces masking


def test_ac10_redaction_surface_present_before_collection_exists(add_turns_harness, monkeypatch):
    """Regression: get_stats keeps the redaction surface on the no-collection path."""
    import contextlib

    rag_engine, _ = add_turns_harness

    class _NoCollectionClient:
        def has_collection(self, name):
            return False

    @contextlib.contextmanager
    def _no_collection_client(db_path=None):
        yield _NoCollectionClient()

    monkeypatch.setattr(rag_engine, "milvus_client", _no_collection_client)
    stats = rag_engine.get_stats(db_path=DB)
    assert "redaction" in stats  # never KeyErrors (AC-10 contract on every path)
    assert stats["redaction"]["enabled"] is True


# === SESF-42 — scan_spans (span-aware, value-free audit scan) ===============
#
# TDD red phase for the new `secret_redaction.scan_spans` companion to `redact`.
# Contract (design.md, Component 1 / Data Models):
#   scan_spans(text, *, mode, allowlist=None) -> tuple[str, list[Span]]
#   Span = NamedTuple(rule_name: str, tier: int, start: int, end: int,
#                     masked_snippet: str)
# Invariants vs redact():
#   * per-OCCURRENCE spans (NO dedup-by-value) with offsets into the ORIGINAL text;
#   * masked_snippet is value-free (the window is redact(..., enforce)-masked);
#   * spans are identical in "report" and "enforce"; only the returned text differs;
#   * deterministic; redact()'s own public contract stays unchanged.
#
# Every token here reuses the part-assembled synthetic fixtures above (AC-18).


def _has_scan_spans():
    """Return True once `secret_redaction.scan_spans` exists (else skip-on-stub)."""
    return hasattr(secret_redaction, "scan_spans")


def test_scan_spans_per_occurrence_no_value_dedup():
    """Two occurrences of the SAME synthetic secret yield TWO spans (no dedup)."""
    scan_spans = secret_redaction.scan_spans
    prefix = "first key "
    middle = " and second key "
    text = f"{prefix}{AWS_KEY}{middle}{AWS_KEY} end"
    _, spans = scan_spans(text, mode="report")
    aws_spans = [s for s in spans if s.rule_name == "AWS"]
    # Two distinct occurrences -> two spans (redact() would dedup to one Hit).
    assert len(aws_spans) == 2
    # Offsets index into the ORIGINAL text and recover the exact value.
    for s in aws_spans:
        assert s.tier == 1
        assert text[s.start:s.end] == AWS_KEY
    # The two spans are at the two real positions, not duplicates of one offset.
    starts = sorted(s.start for s in aws_spans)
    assert starts == [text.find(AWS_KEY), text.find(AWS_KEY, text.find(AWS_KEY) + 1)]


def test_scan_spans_one_span_per_distinct_tier1_token():
    """Each distinct Tier-1 provider token in the text yields its own span."""
    scan_spans = secret_redaction.scan_spans
    text = (
        f"aws {AWS_KEY} gh {GH_KEY} gitlab {GITLAB_KEY} "
        f"openai {OPENAI_KEY} anthropic {ANTHROPIC_KEY}"
    )
    _, spans = scan_spans(text, mode="report")
    rules = sorted(s.rule_name for s in spans)
    for expected in ("AWS", "GITHUB", "GITLAB", "OPENAI", "ANTHROPIC"):
        assert expected in rules
    # Each is tier 1 and its offsets recover the right token.
    by_rule = {s.rule_name: s for s in spans}
    assert by_rule["AWS"].tier == 1
    assert text[by_rule["AWS"].start:by_rule["AWS"].end] == AWS_KEY
    assert text[by_rule["GITHUB"].start:by_rule["GITHUB"].end] == GH_KEY


def test_scan_spans_tier2_assignment_span():
    """A Tier-2 keyword=value assignment yields an ASSIGNMENT / tier-2 span."""
    scan_spans = secret_redaction.scan_spans
    text = f"api_key: {ASSIGN_SECRET_VALUE}"
    _, spans = scan_spans(text, mode="report")
    assign = [s for s in spans if s.rule_name == "ASSIGNMENT"]
    assert assign, "expected an ASSIGNMENT span for the Tier-2 value"
    s = assign[0]
    assert s.tier == 2
    # The span covers the value (not the key); offsets recover the secret value.
    assert text[s.start:s.end] == ASSIGN_SECRET_VALUE


def test_scan_spans_masked_snippet_is_value_free():
    """masked_snippet never contains the synthetic secret substring."""
    scan_spans = secret_redaction.scan_spans
    text = f"my credential is {AWS_KEY} ok"
    _, spans = scan_spans(text, mode="report")
    assert spans
    for s in spans:
        assert AWS_KEY not in s.masked_snippet


def test_scan_spans_masked_snippet_masks_neighbor_secret_in_window():
    """Two synthetic secrets within +-24 chars: BOTH masked in each snippet.

    The ±_SNIPPET_PAD window around one secret may overlap a neighbor; the snippet
    is produced via redact(window, enforce) so the neighbor is masked too.
    """
    scan_spans = secret_redaction.scan_spans
    # GH_KEY immediately after AWS_KEY (separated by a single space) -> well within
    # the ±24-char window, so each secret sits in the other's snippet window.
    text = f"creds {AWS_KEY} {GH_KEY} done"
    _, spans = scan_spans(text, mode="report")
    assert spans
    for s in spans:
        assert AWS_KEY not in s.masked_snippet
        assert GH_KEY not in s.masked_snippet


def test_scan_spans_forced_entropy_snippet_value_free_when_keyword_outside_window():
    """A forced Tier-3 entropy token must not leak RAW in its own snippet (SESF-42).

    Regression: ``_collect_spans`` used to rebuild each snippet by re-running
    ``redact(window, enforce)`` over the +-24-char window. A HIGH_ENTROPY hit is
    only force-masked when a secret keyword is within 40 chars (``_keyword_adjacent``)
    or inside a fence. When the forcing keyword sits >24 but <40 chars from the token,
    the +-24 window EXCLUDES it, so ``redact(window)`` re-evaluated the token as
    un-forced and left it RAW in the snippet -- even though the span was still
    reported. The snippet must be value-free regardless of keyword distance.
    """
    scan_spans = secret_redaction.scan_spans
    token = ENTROPY_FORCED_TOKEN
    # keyword, then ~28 chars of filler, then the token: keyword is within 40 chars
    # (so the token is forced/reported) but outside the token's +-24 snippet window.
    text = "api_key " + "x" * 28 + " " + token
    _, spans = scan_spans(text, mode="report")
    # (1) the token is reported as a forced HIGH_ENTROPY span.
    entropy_spans = [
        s for s in spans
        if s.rule_name == "HIGH_ENTROPY" and text[s.start:s.end] == token
    ]
    assert entropy_spans, "expected a forced HIGH_ENTROPY span for the token"
    # (2) the raw token never appears in ANY span's snippet.
    for s in spans:
        assert token not in s.masked_snippet
    # The target span's own snippet carries the typed placeholder instead.
    assert any("[REDACTED:HIGH_ENTROPY]" in s.masked_snippet for s in entropy_spans)


def test_scan_spans_identical_in_report_and_enforce():
    """Spans are identical across modes; only the returned text differs."""
    scan_spans = secret_redaction.scan_spans
    text = f"cred {AWS_KEY} and {GH_KEY} end"
    report_text, report_spans = scan_spans(text, mode="report")
    enforce_text, enforce_spans = scan_spans(text, mode="enforce")
    # Same audit metadata in both modes.
    assert report_spans == enforce_spans
    # report leaves text unchanged; enforce masks it.
    assert report_text == text
    assert AWS_KEY not in enforce_text
    assert GH_KEY not in enforce_text
    assert "[REDACTED:AWS]" in enforce_text


def test_scan_spans_determinism():
    """Identical input -> identical spans and identical returned text."""
    scan_spans = secret_redaction.scan_spans
    text = f"key {AWS_KEY} then {OPENAI_KEY} done"
    first = scan_spans(text, mode="enforce")
    second = scan_spans(text, mode="enforce")
    assert first == second


def test_scan_spans_span_namedtuple_shape():
    """Span carries exactly rule_name, tier, start, end, masked_snippet."""
    scan_spans = secret_redaction.scan_spans
    _, spans = scan_spans(f"key {AWS_KEY} done", mode="report")
    assert spans
    assert tuple(type(spans[0])._fields) == (
        "rule_name", "tier", "start", "end", "masked_snippet",
    )


def test_scan_spans_allowlist_excludes_value():
    """An operator-allowlisted value yields no span (parity with redact)."""
    import re

    scan_spans = secret_redaction.scan_spans
    text = f"internal {AWS_KEY} ok"
    _, spans = scan_spans(
        text, mode="report", allowlist=[re.compile(re.escape(AWS_KEY))]
    )
    assert not any(s.rule_name == "AWS" for s in spans)
    # Positive control: without the allowlist the AWS span IS present.
    _, spans2 = scan_spans(text, mode="report")
    assert any(s.rule_name == "AWS" for s in spans2)


def test_scan_spans_and_redact_agree_on_masked_values():
    """redact() and scan_spans() agree on every value redact() actually masks.

    Both paths route through `_aggregate_maskable_candidates`, so each value
    redact() removes from enforce-mode output must be reported by scan_spans() at
    the same rule/tier — no behavioral drift between the SESF-41 live guard and
    the SESF-42 retroactive scanner. (redact()'s Hit list additionally carries
    non-forced advisory entropy hits, which scan_spans rightly omits; this test
    compares against the maskable set, the values enforce truly replaces.)
    """
    text = (
        f"aws {AWS_KEY} gh {GH_KEY} api_key: {ASSIGN_SECRET_VALUE} "
        f"issue {ISSUE_ID}"
    )
    redacted, _ = redact(text, mode="enforce")
    _, spans = secret_redaction.scan_spans(text, mode="report")
    # The (value, rule, tier) the masker would replace == what spans report.
    maskable = {
        (text[s.start:s.end], s.rule_name, s.tier) for s in spans
    }
    assert maskable == set(secret_redaction._maskable_values(text, []))
    # Every maskable value is actually gone from redact()'s enforce output.
    for value, _rule, _tier in maskable:
        assert value not in redacted
    # The structurally-exempt issue ID survives and is never reported.
    assert ISSUE_ID in redacted
    assert ISSUE_ID not in {v for v, _r, _t in maskable}


def test_redact_public_contract_unchanged():
    """redact() still returns (str, list[Hit]) with rule_name/tier only."""
    out, hits = redact(f"key {AWS_KEY} done", mode="report")
    assert isinstance(out, str)
    assert isinstance(hits, list)
    assert hits
    assert tuple(type(hits[0])._fields) == ("rule_name", "tier")


# === SESF-44 — Tier-2 ASSIGNMENT precision ==================================
#
# Two FP classes the SESF-42 live baseline surfaced:
#   * the bare `auth` keyword substring-matching the `author*`/`authority`/
#     co-author family (D-A: drop bare `auth`, add a left-anchored auth family);
#   * command-substitution `$(…)` and env-interpolation `${…}` values that carry
#     no literal secret (D-B: reject them in the Tier-2 branch).
# Every case is asserted through BOTH public paths -- `redact()` (Hit) AND
# `scan_spans()` (Span) -- because both route through the shared
# `_aggregate_maskable_candidates` seam (AC-6/AC-7). All tokens are synthetic.

# A synthetic literal value: 16 chars (< _ENTROPY_MIN_LEN=20, so it is NOT a
# Tier-3 entropy candidate -- isolating the Tier-2 behavior under test) and >= 8
# chars with mixed case+digits so it passes `_is_placeholder_value`.
SESF44_LITERAL = "Hunter2" + "Swordfish"


def _has_tier2_hit(line):
    """True if `redact` reports a Tier-2 Hit for `line`."""
    _, hits = redact(line, mode="enforce")
    return any(h.tier == 2 for h in hits)


def _has_assignment_span(line):
    """True if `scan_spans` reports an ASSIGNMENT span for `line`."""
    _, spans = secret_redaction.scan_spans(line, mode="report")
    return any(s.rule_name == "ASSIGNMENT" for s in spans)


# AC-1: the full reconciled author*/authority/co-author family. Each is flagged
# on current `main` (bare `auth` substring) -> RED until D-1 lands.
SESF44_AUTHOR_FAMILY = [
    "author",
    "authored",
    "authoredDate",
    "authorAssociation",
    "authority",
    "gradingAuthority",
    "coauthor",
    "coAuthoredBy",
]


@pytest.mark.parametrize("key", SESF44_AUTHOR_FAMILY)
def test_sesf44_ac1_author_family_not_flagged(key):
    """AC-1: author*/authority/co-author keys are NOT Tier-2 flagged (both paths).

    Value is an 8+ char literal that passes the value guard, so suppression is
    proven by KEY shape (not value length). RED on current `main`.
    """
    line = f'{{"{key}": "{SESF44_LITERAL}"}}'
    out, hits = redact(line, mode="enforce")
    assert not any(h.tier == 2 for h in hits), f"{key} should not be a Tier-2 hit"
    assert SESF44_LITERAL in out  # value left intact (not masked)
    assert not _has_assignment_span(line), f"{key} should yield no ASSIGNMENT span"


def test_sesf44_ac3_command_substitution_not_flagged():
    """AC-3: a `$(…)` command-substitution value is not flagged (both paths).

    `export FOO_API_KEY=$(security …)` captures value `$(security` -- flagged on
    `main` (passes the value guard) -> RED until D-2's value-start check lands.
    """
    line = "export FOO_API_KEY=$(security find-generic-password svc)"
    out, hits = redact(line, mode="enforce")
    assert not any(h.tier == 2 for h in hits)
    assert "$(security" in out  # command-sub left intact
    assert not _has_assignment_span(line)


def test_sesf44_ac4_env_interpolation_not_flagged():
    """AC-4: a `${VAR:-default}` interpolation is not flagged (both paths).

    `x=${FOO_PASSWORD:-default}` matches key `FOO_PASSWORD`, value `-default` --
    the `${` precedes the key, so a value-shape check alone misses it. RED until
    D-2's interpolation-span check lands.
    """
    line = "x=${FOO_PASSWORD:-default}"
    out, hits = redact(line, mode="enforce")
    assert not any(h.tier == 2 for h in hits)
    assert not _has_assignment_span(line)
    # Sanity: a pure `${VAR}` value (no default) is already unflagged (the value
    # capture stops at `{`, leaving a sub-minimum value) -- must stay unflagged.
    pure = "MY_SECRET=${VALUE}"
    assert not _has_tier2_hit(pure)
    assert not _has_assignment_span(pure)


# AC-2 / AC-5: genuine secret-bearing keys with a literal value STILL flag.
# Bare `oauth` is required explicitly -- it is the exact key the D-1 left
# lookbehind targets. These are positive controls: green on `main` AND after C.
SESF44_TRUE_POSITIVE_KEYS = [
    "authorization",
    "oauth",        # bare -- D-1 lookbehind probe
    "oauth_token",
    "auth_token",
    "auth-token",
    "authToken",
    "FOO_API_KEY",
]


@pytest.mark.parametrize("key", SESF44_TRUE_POSITIVE_KEYS)
def test_sesf44_ac2_ac5_genuine_keys_still_flagged(key):
    """AC-2/AC-5: genuine secret keys with a literal value STILL flag (both paths)."""
    line = f'{key}="{SESF44_LITERAL}"'
    out, hits = redact(line, mode="enforce")
    assert any(h.tier == 2 for h in hits), f"{key} should still be a Tier-2 hit"
    assert SESF44_LITERAL not in out  # value masked
    assert _has_assignment_span(line), f"{key} should still yield an ASSIGNMENT span"


def test_sesf44_ac2_placeholder_value_still_skipped():
    """AC-2 consistency: a genuine key with a placeholder value stays unflagged."""
    line = "authorization=none"
    assert not _has_tier2_hit(line)
    assert not _has_assignment_span(line)
