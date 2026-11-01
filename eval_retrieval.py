"""Retrieval eval for a running SessionFlow server.

Runs a labelled JSONL query set through ``search_all_sessions`` or ``search_memories``
over the local MCP endpoint and scores recall@5, MRR and nDCG@10 per query group.
Also measures memory index freshness with a canary memory and rolls up the opt-in
search usage log (``search_log``). Standard library only; it never embeds.

Subcommands: ``run``, ``sweep``, ``freshness``, ``usage``, ``nightly``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import posixpath
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

DEFAULT_URL = os.environ.get("SESSIONFLOW_URL", "http://127.0.0.1:7102/mcp/")
PROTOCOL_VERSION = "2025-06-18"
LAUNCH_AGENT_LABEL = "cc.lbruton.sessionflow"

GROUPS = ("identifier", "paraphrase", "buried-fact", "temporal", "cross-project",
          "no-answer", "superseded")
TOOLS = ("search_all_sessions", "search_memories")
SORT_BY = ("hybrid", "relevance", "recency")
FETCH_N = 10
CANARY_TOPIC = "eval:freshness"
CANARY_PROJECT = "sessionflow"
REFORMULATION_WINDOW_S = 60
# MCP clientInfo name of this harness; the usage rollup skips its calls.
EVAL_CLIENT_NAME = "eval-retrieval"

# Ranking env vars the server reports under /health "ranking", and their keys there.
RANKING_ENV = {
    "SESSIONFLOW_RECENCY_WEIGHT": ("recency_weight", float),
    "SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT": ("temporal_recency_weight", float),
    "SESSIONFLOW_RECENCY_DECAY_DAYS": ("recency_decay_days", int),
}
METRICS = (("recall_at_5", "recall@5"), ("mrr", "MRR"), ("ndcg_at_10", "nDCG@10"))


class EvalError(Exception):
    """A server call, query-set line or setting the eval cannot use."""


# --- Query set -----------------------------------------------------------------


def load_queries(path: Path) -> list[dict]:
    """Read and check a JSONL query set; blank lines are skipped.

    Raises:
        ValueError: a line is not JSON or breaks the schema; the message names the line.
    """
    queries, seen = [], set()
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            q = json.loads(raw)
        except ValueError:
            raise ValueError(f"line {number}: not JSON") from None
        problem = _query_problem(q)
        if problem is None and q["id"] in seen:
            problem = f"duplicate id {q['id']!r}"
        if problem:
            raise ValueError(f"line {number}: {problem}")
        seen.add(q["id"])
        queries.append(q)
    return queries


def _query_problem(q) -> Optional[str]:
    if not isinstance(q, dict):
        return "not an object"
    for key in ("id", "query"):
        if not isinstance(q.get(key), str) or not q[key].strip():
            return f"{key} must be a non-empty string"
    if q.get("group") not in GROUPS:
        return f"group must be one of {', '.join(GROUPS)}"
    if q.get("tool") not in TOOLS:
        return f"tool must be one of {', '.join(TOOLS)}"
    if not isinstance(q.get("args", {}), dict):
        return "args must be an object"
    if "exclude_roots" in q:
        roots = q["exclude_roots"]
        if not isinstance(roots, list) or not roots or not all(
                isinstance(r, str) and r.startswith("/") for r in roots):
            return "exclude_roots must be a non-empty list of absolute paths"
    expected = q.get("expected")
    if not isinstance(expected, list) or not all(isinstance(e, str) for e in expected):
        return "expected must be a list of IDs"
    if (q["group"] == "no-answer") != (not expected):
        return "expected is empty for no-answer queries and only for them"
    if q["group"] == "superseded" and not q.get("superseded"):
        return "superseded queries list the IDs they overturn in superseded"
    return None


# --- Scoring -------------------------------------------------------------------


def result_ids(tool: str, payload: dict) -> list[str]:
    """Return result IDs in rank order: memory IDs, or ``session_id:turn_index``."""
    rows = payload.get("results") or []
    if tool == "search_memories":
        return [str(r.get("memory_id")) for r in rows]
    return [f"{r.get('session_id')}:{r.get('turn_index')}" for r in rows]


def drop_excluded(tool: str, payload: dict, roots: list[str],
                  fill: Optional[dict] = None) -> tuple[dict, int]:
    """Drop rows from an excluded root or below it, then keep ``FETCH_N``.

    Turns carry ``project_root`` and memories ``source_project_root``. Paths are
    normalized and match whole components, so ``/x/SessionFlow`` keeps ``/x/SessionFlowOld``;
    a row without an absolute root never matches. Rows of ``fill`` not already kept are
    appended in order up to ``FETCH_N``.

    Returns:
        The payload with the kept rows, and the number of ``payload`` rows dropped.
    """
    prefixes = [posixpath.normpath(r).rstrip("/") + "/" for r in roots]
    field = "source_project_root" if tool == "search_memories" else "project_root"

    def excluded(row: dict) -> bool:
        root = row.get(field)
        if not isinstance(root, str) or not root.startswith("/"):
            return False
        root = posixpath.normpath(root).rstrip("/") + "/"
        return any(root.startswith(p) for p in prefixes)

    rows = payload.get("results") or []
    kept = [r for r in rows if not excluded(r)]
    dropped = len(rows) - len(kept)
    if fill is not None:
        seen = set(result_ids(tool, {"results": kept}))
        for r in fill.get("results") or []:
            rid = result_ids(tool, {"results": [r]})[0]
            if len(kept) < FETCH_N and rid not in seen and not excluded(r):
                kept.append(r)
                seen.add(rid)
    return {**payload, "results": kept[:FETCH_N]}, dropped


def top_score(tool: str, payload: dict) -> Optional[float]:
    """Return the top transcript hit's cosine similarity; memories carry no score."""
    rows = payload.get("results") or []
    if tool != "search_all_sessions" or not rows or rows[0].get("distance") is None:
        return None
    return 1.0 - float(rows[0]["distance"])


def score_query(q: dict, ranked: list[str], top_score: Optional[float] = None) -> dict:
    """Score one query's ranked IDs against its expected IDs.

    A no-answer query has no recall, MRR or nDCG; its top hit score shows how
    clearly the server could report no match.
    """
    expected = list(dict.fromkeys(q["expected"]))
    wanted = set(expected)
    rank = next((i for i, rid in enumerate(ranked, 1) if rid in wanted), None)
    scored = {
        "id": q["id"], "group": q["group"], "tool": q["tool"], "query": q["query"],
        "expected": expected, "ranked": ranked[:FETCH_N], "rank": rank,
        "recall_at_5": None, "rr": None, "ndcg_at_10": None,
        "top_score": top_score, "current_first": None,
    }
    if q.get("superseded"):
        scored["superseded"] = list(q["superseded"])
    if expected:
        scored["recall_at_5"] = len(wanted & set(ranked[:5])) / len(wanted)
        scored["rr"] = 1.0 / rank if rank else 0.0
        scored["ndcg_at_10"] = _ndcg(ranked[:10], wanted)
    if q.get("superseded"):
        old = set(q["superseded"])
        old_rank = next((i for i, rid in enumerate(ranked, 1) if rid in old), None)
        scored["current_first"] = rank is not None and (old_rank is None or rank < old_rank)
    return scored


def _ndcg(ranked: list[str], wanted: set) -> float:
    seen, dcg = set(), 0.0
    for i, rid in enumerate(ranked):
        if rid in wanted and rid not in seen:
            seen.add(rid)
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(wanted), 10)))
    return dcg / ideal if ideal else 0.0


def _mean(values: Iterable) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _group_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "recall_at_5": _mean(r["recall_at_5"] for r in rows),
        "mrr": _mean(r["rr"] for r in rows),
        "ndcg_at_10": _mean(r["ndcg_at_10"] for r in rows),
        "current_first": _mean(r.get("current_first") for r in rows),
        "top_score": _mean(r.get("top_score") for r in rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "dropped": sum(r.get("dropped") or 0 for r in rows),
    }


def summarize(scored: list[dict]) -> dict:
    """Return per-group means, plus ``all``; a metric with no values is None."""
    out = {}
    for group in GROUPS:
        rows = [r for r in scored if r["group"] == group]
        if rows:
            out[group] = _group_summary(rows)
    out["all"] = _group_summary(scored)
    return out


def build_report(scored: list[dict], run: dict) -> dict:
    """Return the result document: run settings, group summaries and every query."""
    return {"run": {**run, "query_count": len(scored)}, "groups": summarize(scored),
            "queries": scored}


# --- Reports -------------------------------------------------------------------


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(report: dict, title: str = "Retrieval eval") -> str:
    """Return a Markdown summary table of one report."""
    run = report["run"]
    lines = [f"## {title}", ""]
    for key in ("started_at", "server_version", "host", "sort_by", "queries_file", "query_count"):
        if run.get(key) is not None:
            lines.append(f"- {key}: {run[key]}")
    if run.get("env"):
        lines.append("- env: " + ", ".join(f"{k}={v}" for k, v in run["env"].items()))
    if run.get("ranking"):
        lines.append("- ranking: " + ", ".join(f"{k}={v}" for k, v in run["ranking"].items()))
    if run.get("exclude_roots"):
        lines.append("- exclude_roots: " + ", ".join(run["exclude_roots"]))
    lines += ["", "| group | n | recall@5 | MRR | nDCG@10 | current first | top score | errors |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for group, s in report["groups"].items():
        lines.append(f"| {group} | {s['n']} | {_fmt(s['recall_at_5'])} | {_fmt(s['mrr'])} | "
                     f"{_fmt(s['ndcg_at_10'])} | {_fmt(s['current_first'])} | "
                     f"{_fmt(s['top_score'])} | {s['errors']} |")
    misses = [q for q in report["queries"] if q["expected"] and not q["rank"]]
    if misses:
        lines += ["", "Misses: " + ", ".join(q["id"] for q in misses)]
    dropped = [q for q in report["queries"] if q.get("dropped")]
    if dropped:
        lines += ["", "Dropped rows: " + ", ".join(f"{q['id']} {q['dropped']}" for q in dropped)]
    short = [q["id"] for q in report["queries"] if q.get("short")]
    if short:
        lines += ["", "Short after exclusion: " + ", ".join(short)]
    return "\n".join(lines) + "\n"


def render_comparison(reports: dict, metric: str = "mrr") -> str:
    """Return one Markdown table of ``metric`` per group, one column per labelled report."""
    labels = list(reports)
    groups = [g for g in (*GROUPS, "all") if any(g in r["groups"] for r in reports.values())]
    lines = ["| group | " + " | ".join(labels) + " |",
             "| --- |" + " --- |" * len(labels)]
    for group in groups:
        cells = [_fmt(reports[label]["groups"].get(group, {}).get(metric)) for label in labels]
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` into ``out_dir`` and return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report, title=stem), encoding="utf-8")
    return json_path, md_path


# --- Ranking settings ----------------------------------------------------------


def parse_env(pairs: Iterable[str]) -> dict:
    """Turn repeated ``KEY=VALUE`` arguments into a dict.

    Raises:
        ValueError: an argument has no ``=``.
    """
    env = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def check_ranking(env: dict, health: dict) -> tuple[list[str], list[str]]:
    """Compare the expected env with the server's /health ``ranking``.

    Returns:
        ``(problems, unverified)``: mismatches that make the run invalid, and env
        names the server does not report, which the run records unchecked.
    """
    problems, unverified = [], []
    ranking = health.get("ranking")
    for name, value in env.items():
        if name not in RANKING_ENV:
            unverified.append(name)
            continue
        if not isinstance(ranking, dict):
            problems.append("the server reports no ranking settings in /health; restart it on this version")
            continue
        key, cast = RANKING_ENV[name]
        try:
            want = cast(value)
        except ValueError:
            problems.append(f"{name}={value} is not a valid {cast.__name__}")
            continue
        have = ranking.get(key)
        if have is None or not math.isclose(float(have), float(want), abs_tol=1e-9):
            problems.append(f"{name}={value} expected, but the server applies {key}={have}")
    return problems, unverified


# --- MCP client ----------------------------------------------------------------


class McpClient:
    """Stateful MCP client for the SessionFlow HTTP endpoint (initialize, call, delete)."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 60.0):
        """Store the endpoint; a base URL such as ``http://127.0.0.1:7102`` gets ``/mcp/``."""
        parts = urllib.parse.urlsplit(url)
        if not parts.path.rstrip("/"):
            url = urllib.parse.urlunsplit(parts._replace(path="/mcp/"))
        self.url = url
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._next_id = 0

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        return headers

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            ctype = resp.headers.get("Content-Type", "")
            if resp.headers.get("Mcp-Session-Id"):
                self.session_id = resp.headers["Mcp-Session-Id"]
        if not raw:
            return {}
        if ctype.startswith("text/event-stream"):
            events = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
            raw = next((e for e in reversed(events) if e), "")
        try:
            return json.loads(raw)
        except ValueError:
            raise EvalError("the server reply is not JSON") from None

    def _request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        return self._post({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})

    def _open(self) -> None:
        self.session_id = None
        self._request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                     "clientInfo": {"name": EVAL_CLIENT_NAME, "version": "1"}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, name: str, arguments: dict) -> dict:
        """Call one tool with ``format: json`` and return its parsed JSON payload.

        A 404 means the server dropped the session (restart or idle expiry); the
        client opens a new one and retries once.

        Raises:
            EvalError: the server is unreachable or the tool returned an error.
        """
        params = {"name": name, "arguments": {**arguments, "format": "json"}}
        for attempt in (1, 2):
            try:
                if not self.session_id:
                    self._open()
                reply = self._request("tools/call", params)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and attempt == 1:
                    self.session_id = None
                    continue
                raise EvalError(f"{name}: HTTP {exc.code}") from None
            except (urllib.error.URLError, OSError) as exc:
                raise EvalError(f"{name}: server unreachable ({getattr(exc, 'reason', exc)})") from None
        if "error" in reply:
            raise EvalError(f"{name}: JSON-RPC error {reply['error'].get('message')}")
        result = reply.get("result") or {}
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        if result.get("isError"):
            raise EvalError(f"{name}: {text}")
        try:
            return json.loads(text)
        except ValueError:
            raise EvalError(f"{name}: {text.strip()[:200]}") from None

    def health(self) -> dict:
        """Return the /health JSON, also while the server answers 503 during startup."""
        url = self.url.split("/mcp", 1)[0] + "/health"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read())
            except ValueError:
                raise EvalError(f"/health: HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise EvalError(f"/health: {getattr(exc, 'reason', exc)}") from None

    def close(self) -> None:
        """Ask the server to drop the session; it expires idle sessions anyway."""
        if not self.session_id:
            return
        req = urllib.request.Request(self.url, method="DELETE", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, OSError):
            pass
        self.session_id = None


def _host(health: dict) -> str:
    return ((health.get("memory") or {}).get("index") or {}).get("host") or socket.gethostname()


# --- Query runs ----------------------------------------------------------------


def run_queries(client, queries: list[dict], sort_by: str) -> list[dict]:
    """Run every query and score it; a failed call scores as a miss and keeps its error.

    A query with ``exclude_roots`` drops matching rows from the normal ``FETCH_N`` and
    fills the gap from a ``FETCH_N * 3`` search. The fill comes second because ``n``
    changes the server's candidate pool and so the order of the top rows. ``short``
    marks a query whose full fill search still left fewer than ``FETCH_N`` rows.
    """
    scored = []
    for q in queries:
        roots = q.get("exclude_roots")
        args = {**q.get("args", {}), "query": q["query"], "n": FETCH_N, "sort_by": sort_by}
        dropped, short = 0, False
        try:
            payload = client.call(q["tool"], args)
            if roots:
                payload, dropped = drop_excluded(q["tool"], payload, roots)
                if dropped:
                    wider = client.call(q["tool"], {**args, "n": FETCH_N * 3})
                    payload, _ = drop_excluded(q["tool"], payload, roots, fill=wider)
                    short = (len(payload["results"]) < FETCH_N
                             and len(wider.get("results") or []) >= FETCH_N * 3)
            row = score_query(q, result_ids(q["tool"], payload), top_score(q["tool"], payload))
        except EvalError as exc:
            row = score_query(q, [])
            row["error"] = str(exc)
        if roots:
            row["dropped"] = dropped
            row["short"] = short
        scored.append(row)
    return scored


def evaluate(client, queries: list[dict], *, sort_by: str, env: dict, queries_file: str) -> dict:
    """Check the server's ranking against ``env``, run the set and return the report.

    Raises:
        EvalError: the server applies different ranking settings than ``env`` names.
    """
    health = client.health()
    problems, unverified = check_ranking(env, health)
    if problems:
        raise EvalError("; ".join(problems))
    started = datetime.now(timezone.utc).isoformat()
    scored = run_queries(client, queries, sort_by)
    run = {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
           "server_version": health.get("version"), "host": _host(health), "sort_by": sort_by,
           "env": env, "env_unverified": unverified, "ranking": health.get("ranking"),
           "queries_file": queries_file,
           "exclude_roots": sorted({r for q in queries for r in q.get("exclude_roots", [])})}
    return build_report(scored, run)


# --- Freshness probe -----------------------------------------------------------


def make_token() -> str:
    """Return a unique canary token: letters and digits, one FTS term, no secret-like entropy."""
    return f"sfcanary{time.time_ns()}{secrets.randbelow(10_000):04d}"


def canary_text(token: str) -> str:
    """Return the canary memory text that carries ``token``."""
    return f"Freshness canary {token}, written by eval_retrieval.py and deleted after the probe."


def default_lock_path() -> Path:
    """Return the file lock that serializes freshness probes on this host."""
    return Path.home() / ".sessionflow" / "eval-freshness.lock"


def freshness_probe(client, *, project: str = CANARY_PROJECT, agent_id: str = "claude-code",
                    poll_s: float = 1.0, timeout_s: float = 120.0,
                    sleep: Callable[[float], None] = time.sleep,
                    clock: Callable[[], float] = time.monotonic,
                    lock_path: Optional[Path] = None) -> dict:
    """Write a canary memory, poll ``search_memories`` until it appears, then delete it.

    The ``topic_key`` names the host, so a later probe there replaces a canary a
    failed delete left behind. ``lock_path`` makes probes on one host wait for each
    other, and a hit counts only when the record still carries this probe's token.
    """
    if lock_path is None:
        return _probe(client, project, agent_id, poll_s, timeout_s, sleep, clock)
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(lock_path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return _probe(client, project, agent_id, poll_s, timeout_s, sleep, clock)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _delete_exact_canary(client, project: str, token: str) -> bool:
    """Delete memories whose text equals this token's canary text; True when one was deleted."""
    text = canary_text(token)
    try:
        found = client.call("search_memories", {"query": token, "project": project, "n": 5})
    except EvalError:
        return False
    deleted = False
    for row in found.get("results") or []:
        if row.get("text") == text:
            try:
                client.call("delete_memory", {"memory_id": row["memory_id"], "project": project})
                deleted = True
            except EvalError:
                pass
    return deleted


def _probe(client, project, agent_id, poll_s, timeout_s, sleep, clock) -> dict:
    token = make_token()
    result = {"host": None, "memory_id": None, "add_outcome": None, "add_index": None,
              "time_to_hit_s": None, "polls": 0, "channels": None, "health": None,
              "deleted": False}
    start = clock()
    try:
        added = client.call("add_memory", {
            "text": canary_text(token), "agent_id": agent_id, "project": project, "type": "fact",
            "topic_key": f"{CANARY_TOPIC}:{socket.gethostname()}", "extra": {"source": "eval"}})
    except EvalError as exc:
        # The add may have committed before the error; delete only an exact match.
        result["error"] = str(exc)
        result["deleted"] = _delete_exact_canary(client, project, token)
        return result
    result.update(memory_id=added.get("memory_id"), add_outcome=added.get("outcome"),
                  add_index=added.get("index"))
    try:
        try:
            health = client.health()
            memory = health.get("memory") or {}
            index = memory.get("index") or {}
            result["host"] = _host(health)
            result["health"] = {"ack_lag": memory.get("ack_lag"),
                                "last_tick_age_s": index.get("last_tick_age_s"),
                                "stale": index.get("stale")}
        except EvalError as exc:
            result["host"] = socket.gethostname()
            result["health_error"] = str(exc)
        while True:
            result["polls"] += 1
            try:
                found = client.call("search_memories", {"query": token, "project": project, "n": 5})
            except EvalError as exc:
                found, result["search_error"] = {}, str(exc)
            if any(r.get("memory_id") == result["memory_id"] and token in (r.get("text") or "")
                   for r in found.get("results") or []):
                result["time_to_hit_s"] = clock() - start
                result["channels"] = found.get("channels")
                break
            if clock() - start >= timeout_s:
                break
            sleep(poll_s)
    finally:
        try:
            client.call("delete_memory", {"memory_id": result["memory_id"], "project": project})
            result["deleted"] = True
        except EvalError as exc:
            result["delete_error"] = str(exc)
    return result


def render_freshness(result: dict) -> str:
    """Return a Markdown summary of one freshness probe."""
    hit = result.get("time_to_hit_s")
    lines = ["## Freshness", "",
             f"- host: {result.get('host')}",
             f"- time to hit: {'timed out' if hit is None else f'{hit:.1f} s'} "
             f"after {result.get('polls')} polls",
             f"- add index outcome: {result.get('add_index')}",
             f"- channels at hit: {', '.join(result.get('channels') or []) or '-'}",
             f"- health: {json.dumps(result.get('health'))}",
             f"- canary deleted: {result.get('deleted')}"]
    for key in ("error", "health_error", "search_error", "delete_error"):
        if result.get(key):
            lines.append(f"- {key}: {result[key]}")
    return "\n".join(lines) + "\n"


# --- Usage rollup --------------------------------------------------------------


def read_usage_log(path: Path) -> list[dict]:
    """Return the usage log entries; unreadable lines are skipped."""
    entries = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return entries
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "ts" in entry:
            entries.append(entry)
    return entries


def usage_rollup(entries: list[dict], *, now: datetime, days: int = 7) -> dict:
    """Roll up the usage log over the last ``days``.

    A follow-up counts as top 3 when it opens one of the first three IDs of the
    same client's latest search. A search counts as reformulated when the same
    client searches again within 60 s with no follow-up in between. Calls from
    this harness (client ``eval-retrieval:...``) are skipped and counted.
    """
    cutoff = now - timedelta(days=days)
    timed, skipped = [], 0
    for e in entries:
        if str(e.get("client") or "").startswith(f"{EVAL_CLIENT_NAME}:"):
            skipped += 1
            continue
        try:
            ts = datetime.fromisoformat(e["ts"])
        except (TypeError, ValueError):
            continue
        if ts >= cutoff:
            timed.append((ts, e))
    timed.sort(key=lambda pair: pair[0])
    latest: dict = {}
    searches = follow_ups = top3 = reformulated = 0
    for ts, e in timed:
        client = e.get("client")
        prev = latest.get(client)
        if e.get("kind") == "search":
            searches += 1
            if prev and not prev["followed"] and \
                    (ts - prev["ts"]).total_seconds() <= REFORMULATION_WINDOW_S:
                reformulated += 1
            latest[client] = {"ts": ts, "ids": list(e.get("ids") or []), "followed": False}
        elif e.get("kind") == "follow_up":
            follow_ups += 1
            if prev:
                prev["followed"] = True
                if e.get("id") in prev["ids"][:3]:
                    top3 += 1
    return {
        "days": days, "searches": searches, "follow_ups": follow_ups,
        "follow_up_top3": top3, "reformulated_searches": reformulated,
        "follow_up_top3_share": top3 / follow_ups if follow_ups else None,
        "reformulation_share": reformulated / searches if searches else None,
        "eval_calls_skipped": skipped,
    }


# --- Server restarts for sweeps ------------------------------------------------


def _launchctl(*args: str) -> str:
    done = subprocess.run(["launchctl", *args], check=True, capture_output=True, text=True)
    return done.stdout


def _launchctl_getenv(name: str) -> Optional[str]:
    """Return the launchd value of ``name``, or None when it is unset."""
    value = _launchctl("getenv", name).strip()
    return value or None


def restore_server(var: str, prior: Optional[str], label: str, url: str) -> list[str]:
    """Put ``var`` back to its value before the sweep, restart and wait; return the failures.

    Each step runs even when an earlier one fails, so a failed ``setenv`` still
    leaves a restarted server.
    """
    errors = []
    try:
        if prior is None:
            _launchctl("unsetenv", var)
        else:
            _launchctl("setenv", var, prior)
    except (subprocess.CalledProcessError, OSError) as exc:
        errors.append(f"launchctl env for {var}: {exc}")
    try:
        _restart_server(label)
    except (subprocess.CalledProcessError, OSError) as exc:
        errors.append(f"restart {label}: {exc}")
    expected = {var: prior} if prior is not None and var in RANKING_ENV else {}
    try:
        wait_ready(McpClient(url), expected)
    except EvalError as exc:
        errors.append(str(exc))
    return errors


def _restart_server(label: str) -> None:
    _launchctl("kickstart", "-k", f"gui/{os.getuid()}/{label}")


def wait_ready(client, env: dict, timeout_s: float = 300.0) -> dict:
    """Wait until /health is ok and applies ``env``; return the health JSON.

    Raises:
        EvalError: the server is not ready with ``env`` within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    last = "no answer"
    while time.monotonic() < deadline:
        try:
            health = client.health()
            problems, _ = check_ranking(env, health)
            if health.get("status") == "ok" and not problems:
                return health
            last = "; ".join(problems) or f"status {health.get('status')}"
        except EvalError as exc:
            last = str(exc)
        time.sleep(2)
    raise EvalError(f"the server was not ready within {timeout_s:.0f} s: {last}")


# --- CLI -----------------------------------------------------------------------


def _local_date() -> str:
    return datetime.now().astimezone().date().isoformat()


def _cmd_run(args) -> int:
    queries = load_queries(args.queries)
    env = parse_env(args.env)
    client = McpClient(args.url)
    try:
        for sort_by in args.sort_by:
            report = evaluate(client, queries, sort_by=sort_by, env=env, queries_file=str(args.queries))
            stem = f"{_local_date()}-{sort_by}" + "".join(f"-{k}={v}" for k, v in env.items())
            if args.out:
                json_path, _ = write_report(report, args.out, stem)
                print(f"Wrote {json_path}", file=sys.stderr)
            print(render_markdown(report, title=stem))
    finally:
        client.close()
    return 0


def _cmd_sweep(args) -> int:
    if not args.yes:
        print(f"sweep restarts the {args.label} LaunchAgent once per value and once to restore it. "
              "Pass --yes to run it.", file=sys.stderr)
        return 2
    queries = load_queries(args.queries)
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    prior = _launchctl_getenv(args.var)
    reports = {}
    client = McpClient(args.url)
    cleanup_errors: list[str] = []
    try:
        for value in values:
            env = {args.var: value}
            _launchctl("setenv", args.var, value)
            _restart_server(args.label)
            wait_ready(client, env)
            client.session_id = None
            report = evaluate(client, queries, sort_by=args.sort_by, env=env,
                              queries_file=str(args.queries))
            label = f"{args.var}={value}"
            reports[label] = report
            write_report(report, args.out, f"{_local_date()}-sweep-{args.sort_by}-{label}")
    finally:
        client.close()
        cleanup_errors = restore_server(args.var, prior, args.label, args.url)
        for error in cleanup_errors:
            print(f"eval_retrieval: restore failed: {error}", file=sys.stderr)
    parts = [f"## Sweep of {args.var} ({args.sort_by})", ""]
    for metric, name in METRICS:
        parts += [f"### {name}", "", render_comparison(reports, metric)]
    text = "\n".join(parts)
    out = Path(args.out) / f"{_local_date()}-sweep-{args.sort_by}-{args.var}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}", file=sys.stderr)
    return 1 if cleanup_errors else 0


def _cmd_freshness(args) -> int:
    client = McpClient(args.url)
    try:
        result = freshness_probe(client, timeout_s=args.timeout, lock_path=default_lock_path())
    finally:
        client.close()
    print(render_freshness(result))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        path = Path(args.out) / f"{_local_date()}-freshness.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}", file=sys.stderr)
    return 0 if result.get("deleted") and result.get("time_to_hit_s") is not None else 1


def _cmd_usage(args) -> int:
    import search_log

    path = args.log or search_log.log_path()
    roll = usage_rollup(read_usage_log(path), now=datetime.now(timezone.utc), days=args.days)
    print(json.dumps(roll, indent=2))
    return 0


def _cmd_nightly(args) -> int:
    queries = load_queries(args.queries)
    client = McpClient(args.url)
    reports, status = {}, 0
    try:
        for sort_by in ("hybrid", "relevance"):
            reports[sort_by] = evaluate(client, queries, sort_by=sort_by, env={},
                                        queries_file=str(args.queries))
        freshness = freshness_probe(client, lock_path=default_lock_path())
    finally:
        client.close()
    if not freshness.get("deleted") or freshness.get("time_to_hit_s") is None:
        status = 1
    stem = f"{_local_date()}-nightly"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"date": _local_date(), "reports": reports, "freshness": freshness}
    (out / f"{stem}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    md = [f"# Nightly retrieval eval {_local_date()}", ""]
    md += [render_markdown(r, title=f"Queries, {sort_by}") for sort_by, r in reports.items()]
    md.append(render_freshness(freshness))
    (out / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out / stem}.json and .md", file=sys.stderr)
    return status


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="MCP endpoint (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Score a query set against the running server.")
    run.add_argument("--queries", type=Path, required=True)
    run.add_argument("--sort-by", choices=SORT_BY, action="append",
                     help="Ranking to score; repeat for several (default: hybrid).")
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                     help="Ranking env the server must apply; the run refuses on a mismatch.")
    run.add_argument("--out", type=Path, help="Write <date>-<sort_by>.json and .md here.")
    run.set_defaults(func=_cmd_run)

    sweep = sub.add_parser("sweep", help="Restart the server once per env value and compare.")
    sweep.add_argument("--queries", type=Path, required=True)
    sweep.add_argument("--var", required=True, help="Env var to sweep, for example SESSIONFLOW_RECENCY_WEIGHT.")
    sweep.add_argument("--values", required=True, help="Comma-separated values, for example 0,0.15,0.3,0.5.")
    sweep.add_argument("--sort-by", choices=SORT_BY, default="hybrid")
    sweep.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--label", default=LAUNCH_AGENT_LABEL, help="Server LaunchAgent label.")
    sweep.add_argument("--yes", action="store_true", help="Allow the server restarts.")
    sweep.set_defaults(func=_cmd_sweep)

    fresh = sub.add_parser("freshness", help="Time a canary memory until search finds it.")
    fresh.add_argument("--timeout", type=float, default=120.0)
    fresh.add_argument("--out", type=Path)
    fresh.set_defaults(func=_cmd_freshness)

    usage = sub.add_parser("usage", help="Roll up the search usage log.")
    usage.add_argument("--days", type=int, default=7)
    usage.add_argument("--log", type=Path, help="Log file (default: ~/.sessionflow/search-log.jsonl).")
    usage.set_defaults(func=_cmd_usage)

    nightly = sub.add_parser("nightly", help="Run hybrid, relevance and freshness into one dated report.")
    nightly.add_argument("--queries", type=Path, required=True)
    nightly.add_argument("--out", type=Path, required=True)
    nightly.set_defaults(func=_cmd_nightly)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the CLI and return its exit code."""
    args = build_parser().parse_args(argv)
    if getattr(args, "sort_by", None) is None and args.command == "run":
        args.sort_by = ["hybrid"]
    try:
        return args.func(args)
    except (EvalError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"eval_retrieval: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
