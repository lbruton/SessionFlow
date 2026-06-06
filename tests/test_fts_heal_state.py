"""Tests for SESF-38 FtsHealState backoff + log-once dedup (AC-2, AC-5).

These tests verify ``http_server.FtsHealState``: ``record_failure`` /
``record_success`` track the failure streak, ``next_delay`` escalates and caps,
and ``should_warn`` deduplicates repeated error signatures. They specify the
implemented behavior described below.

AC-2 (bounded backoff, never abandon): consecutive transient failures escalate
the delay as ``base * 2**(n-1)`` capped at a maximum and held at the cap; the
schedule never returns a terminal "give up" sentinel. A success resets the
streak and the delay back to the base interval (the drain interval, 30s).

AC-5 (log-once dedup): ``should_warn`` returns ``True`` on the first sighting of
an error signature and whenever the signature changes, ``False`` while the same
signature repeats. The signature is derived from the exception type plus a
*stable* message prefix, so a volatile suffix (request id, timestamp) does not
churn the latch. A success resets the latch so a later failure warns again.
"""

import pytest

import http_server

try:  # Prefer exercising the real Milvus exception signature logic.
    from pymilvus.exceptions import MilvusException
except Exception:  # pragma: no cover - pymilvus is a hard dep, but stay defensive.
    MilvusException = None


BASE = 30.0  # FtsHealState base interval defaults to the drain interval (30s).


def _milvus_exc(message: str):
    """Build a MilvusException with ``message``, falling back to RuntimeError."""
    if MilvusException is not None:
        try:
            return MilvusException(message=message)
        except TypeError:
            return MilvusException(0, message)
    return RuntimeError(message)


# --------------------------------------------------------------------------- #
# AC-2 — bounded backoff, never abandon
# --------------------------------------------------------------------------- #


def test_next_delay_escalates_then_plateaus_at_cap():
    """Delays double per consecutive failure (30->60->120->240) then hold at cap."""
    state = http_server.FtsHealState()

    exc = _milvus_exc("transient heal failure")

    state.record_failure(exc)
    assert state.next_delay() == pytest.approx(30.0)

    state.record_failure(exc)
    assert state.next_delay() == pytest.approx(60.0)

    state.record_failure(exc)
    assert state.next_delay() == pytest.approx(120.0)

    state.record_failure(exc)
    assert state.next_delay() == pytest.approx(240.0)

    # 5th failure would be 480; the schedule caps it (e.g. 300) instead.
    state.record_failure(exc)
    capped = state.next_delay()
    assert capped < 480.0, "5th delay must be capped below the raw 2**(n-1) value"
    assert capped == pytest.approx(300.0)

    # Further failures stay at the cap, never escalating without bound.
    for _ in range(5):
        state.record_failure(exc)
        assert state.next_delay() == pytest.approx(capped)


def test_next_delay_is_monotonic_and_never_terminal():
    """Escalation is non-decreasing, finite, positive, and never a give-up sentinel."""
    state = http_server.FtsHealState()
    exc = _milvus_exc("transient heal failure")

    previous = None
    for _ in range(12):
        state.record_failure(exc)
        delay = state.next_delay()
        # Never abandon: a finite, positive, real number every time.
        assert isinstance(delay, (int, float))
        assert delay > 0.0
        assert delay != float("inf")
        assert delay == delay  # not NaN
        if previous is not None:
            assert delay >= previous
        previous = delay


def test_record_success_resets_streak_and_delay_to_base():
    """A success zeroes consecutive_failures and resets next_delay to the base."""
    state = http_server.FtsHealState()
    exc = _milvus_exc("transient heal failure")

    for _ in range(4):
        state.record_failure(exc)
    assert state.consecutive_failures == 4
    assert state.next_delay() == pytest.approx(240.0)

    state.record_success()
    assert state.consecutive_failures == 0

    # After reset, the first subsequent failure starts the schedule over at base.
    state.record_failure(exc)
    assert state.next_delay() == pytest.approx(BASE)


# --------------------------------------------------------------------------- #
# AC-5 — log-once dedup
# --------------------------------------------------------------------------- #


def test_should_warn_first_true_then_false_on_repeat():
    """First sighting warns; identical repeats are suppressed."""
    state = http_server.FtsHealState()
    exc = _milvus_exc("field embedding does not exist")

    assert state.should_warn(exc) is True
    assert state.should_warn(_milvus_exc("field embedding does not exist")) is False
    assert state.should_warn(_milvus_exc("field embedding does not exist")) is False


def test_should_warn_true_when_signature_changes():
    """A different error signature warns again even mid-streak."""
    state = http_server.FtsHealState()

    assert state.should_warn(_milvus_exc("field embedding does not exist")) is True
    # Same signature -> suppressed.
    assert state.should_warn(_milvus_exc("field embedding does not exist")) is False
    # Different message -> new signature -> warns.
    assert state.should_warn(_milvus_exc("collection not loaded")) is True
    # Now that one repeats -> suppressed.
    assert state.should_warn(_milvus_exc("collection not loaded")) is False


def test_should_warn_resets_after_success():
    """record_success() clears the latch so a later failure warns again."""
    state = http_server.FtsHealState()
    exc = _milvus_exc("field embedding does not exist")

    assert state.should_warn(exc) is True
    assert state.should_warn(_milvus_exc("field embedding does not exist")) is False

    state.record_success()

    # Latch cleared: the same signature warns once more after the success.
    assert state.should_warn(_milvus_exc("field embedding does not exist")) is True


def test_signature_ignores_volatile_suffix():
    """Two same-type exceptions differing only by a volatile suffix collapse to one warn."""
    state = http_server.FtsHealState()

    first = _milvus_exc("field embedding does not exist (req-abc123)")
    second = _milvus_exc("field embedding does not exist (req-def456)")

    # Sanity: the raw messages genuinely differ.
    assert str(first) != str(second)

    assert state.should_warn(first) is True
    # Same type + stable prefix -> signature matches -> suppressed despite the
    # different request-id suffix.
    assert state.should_warn(second) is False


def test_signature_distinguishes_exception_type():
    """Different exception types with similar text are distinct signatures."""
    state = http_server.FtsHealState()

    class OtherHealError(RuntimeError):
        """Stand-in distinct error type for signature differentiation."""

    # First sighting of each type warns; same type + message repeating is suppressed.
    assert state.should_warn(_milvus_exc("connection failed")) is True
    # Same type + same message -> suppressed (fails against the always-True stub).
    assert state.should_warn(_milvus_exc("connection failed")) is False
    # Different exception type, identical message -> distinct signature -> warns.
    assert state.should_warn(OtherHealError("connection failed")) is True
    # That new type now repeats -> suppressed.
    assert state.should_warn(OtherHealError("connection failed")) is False
