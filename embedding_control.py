"""Local-only embedding resource controls for SessionFlow backfill work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import os
import time
from typing import Optional

_LOG = logging.getLogger("sessionflow.embedding_budget")

# Default chosen to be effectively non-blocking for a continuous server while
# still bounding accidental runaway jobs. SESF-12 raised this from 200 (which
# silently killed inserts after ~minutes of uptime).
DEFAULT_MAX_TURNS_PER_RUN = 100_000


LOCAL_MODEL_DIMS = {
    "embeddinggemma": 768,
    "modernbert": 768,
}


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(value, minimum)
    return value


def _env_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read a float env var, falling back to ``default`` on bad input.

    Unlike ``_env_int`` (which clamps to ``minimum``), an unparseable value OR a
    value outside ``[minimum, maximum]`` returns ``default`` rather than clamping.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    # NaN/Inf parse cleanly via float() but slip past the bounds checks below
    # (every NaN comparison is False), so reject non-finite values explicitly —
    # NaN would otherwise propagate into hybrid scores and corrupt sort order.
    if not math.isfinite(value):
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class EmbeddingDecision:
    """Outcome of a pre-batch budget check: whether embedding may proceed."""

    allowed: bool
    reason: str = ""
    retry_after_seconds: float = 0.0


@dataclass
class EmbeddingIdentity:
    """Identity of the embedding model/collection a vector set belongs to."""

    embedding_provider: str
    model_name: str
    dimension: int
    collection_name: str = "sessions"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def current_local(cls) -> "EmbeddingIdentity":
        """Build the identity for the locally-configured MLX model."""
        model_name = os.getenv("SESSIONFLOW_MODEL", "embeddinggemma").lower()
        dimension = LOCAL_MODEL_DIMS.get(model_name)
        if dimension is None:
            raise ValueError(
                f"Unknown local embedding model {model_name!r}; "
                f"valid options: {', '.join(sorted(LOCAL_MODEL_DIMS))}"
            )
        return cls(
            embedding_provider="local_mlx",
            model_name=model_name,
            dimension=dimension,
        )


@dataclass
class EmbeddingBudget:
    """Shared throttle and accounting for backfill embedding across providers."""

    batch_size: int = 16
    cooldown_ms: int = 200
    max_turns_per_run: int = DEFAULT_MAX_TURNS_PER_RUN
    max_files_per_run: int = 100
    recent_days: int = 14
    mode: str = "recent"
    paused: bool = False
    turns_processed: int = 0
    batches_processed: int = 0
    errors: int = 0
    last_batch_finished_at: float = 0.0
    last_batch_duration: float = 0.0
    _cap_warned: bool = field(default=False, repr=False)

    @classmethod
    def from_env(cls) -> "EmbeddingBudget":
        """Build a budget from ``SESSIONFLOW_*`` environment overrides."""
        mode = os.getenv("SESSIONFLOW_BACKFILL_MODE", "recent").lower()
        if mode not in {"recent", "incremental", "full"}:
            mode = "recent"
        return cls(
            batch_size=_env_int("SESSIONFLOW_EMBED_BATCH_SIZE", 16, minimum=1),
            cooldown_ms=_env_int("SESSIONFLOW_EMBED_COOLDOWN_MS", 200, minimum=200),
            max_turns_per_run=_env_int(
                "SESSIONFLOW_BACKFILL_MAX_TURNS_PER_RUN",
                DEFAULT_MAX_TURNS_PER_RUN,
                minimum=1,
            ),
            max_files_per_run=_env_int("SESSIONFLOW_BACKFILL_MAX_FILES_PER_RUN", 100, minimum=1),
            recent_days=_env_int("SESSIONFLOW_BACKFILL_RECENT_DAYS", 14, minimum=1),
            mode=mode,
            paused=_env_bool("SESSIONFLOW_BACKFILL_PAUSED", False),
        )

    def before_batch(self, batch_size: int, estimated_chars: int = 0) -> EmbeddingDecision:
        """Decide whether a batch may embed now, honoring pause, cap, and cooldown."""
        if self.paused:
            return EmbeddingDecision(False, "Backfill embedding is paused")
        if self.turns_processed + batch_size > self.max_turns_per_run:
            if not self._cap_warned:
                _LOG.warning(
                    "SESSIONFLOW_BACKFILL_MAX_TURNS_PER_RUN=%d reached after "
                    "%d turns; subsequent batches will be denied until the "
                    "budget is reset or the cap is raised.",
                    self.max_turns_per_run,
                    self.turns_processed,
                )
                self._cap_warned = True
            return EmbeddingDecision(False, "Backfill max turns per run reached")
        if batch_size > self.batch_size:
            return EmbeddingDecision(
                False,
                f"Batch size {batch_size} exceeds configured limit {self.batch_size}",
            )

        elapsed = time.monotonic() - self.last_batch_finished_at
        cooldown = self.cooldown_ms / 1000.0
        if self.last_batch_finished_at and elapsed < cooldown:
            return EmbeddingDecision(
                False,
                "Embedding cooldown active",
                retry_after_seconds=max(0.0, cooldown - elapsed),
            )

        return EmbeddingDecision(True)

    def after_batch(self, duration: float, turns: int, error: Optional[BaseException] = None) -> None:
        """Record a finished batch's duration, turn count, and any error."""
        self.last_batch_finished_at = time.monotonic()
        self.last_batch_duration = duration
        self.batches_processed += 1
        self.turns_processed += turns
        if error is not None:
            self.errors += 1

    def split_batches(self, turns: list) -> list[list]:
        """Split ``turns`` into chunks of at most ``batch_size``."""
        return [turns[i:i + self.batch_size] for i in range(0, len(turns), self.batch_size)]

    def status(self) -> dict:
        """Return a JSON-serializable snapshot of budget config and counters."""
        try:
            model_name = EmbeddingIdentity.current_local().model_name
        except ValueError:
            model_name = "unknown"
        return {
            "embedding_provider": "local_mlx",
            "model_name": model_name,
            "batch_size": self.batch_size,
            "cooldown_ms": self.cooldown_ms,
            "max_turns_per_run": self.max_turns_per_run,
            "max_files_per_run": self.max_files_per_run,
            "recent_days": self.recent_days,
            "mode": self.mode,
            "paused": self.paused,
            "turns_processed": self.turns_processed,
            "batches_processed": self.batches_processed,
            "errors": self.errors,
        }


_BUDGET: Optional[EmbeddingBudget] = None


def get_embedding_budget() -> EmbeddingBudget:
    """Return the process-wide embedding budget, creating it on first use."""
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = EmbeddingBudget.from_env()
    return _BUDGET


def reset_embedding_budget() -> EmbeddingBudget:
    """Rebuild the process-wide embedding budget from the environment."""
    global _BUDGET
    _BUDGET = EmbeddingBudget.from_env()
    return _BUDGET
