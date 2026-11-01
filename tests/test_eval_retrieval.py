"""Retrieval eval harness: query-set loading, metric math, reports, usage rollup, freshness probe."""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import eval_retrieval as ev
import secret_redaction

FIXTURE = Path(__file__).parent / "eval" / "fixture_queries.jsonl"


def _queries():
    return {q["id"]: q for q in ev.load_queries(FIXTURE)}


# --- Query set -----------------------------------------------------------------


def test_fixture_set_loads_every_group():
    queries = ev.load_queries(FIXTURE)
    assert {q["group"] for q in queries} == set(ev.GROUPS)
    assert all(q["tool"] in ev.TOOLS for q in queries)


@pytest.mark.parametrize("line,needle", [
    ({"id": "a", "group": "bogus", "tool": "search_memories", "query": "x", "expected": []}, "group"),
    ({"id": "a", "group": "identifier", "tool": "get_turns", "query": "x", "expected": []}, "tool"),
    ({"id": "a", "group": "identifier", "tool": "search_memories", "query": "x"}, "expected"),
    ({"id": "a", "group": "identifier", "tool": "search_memories", "expected": ["m"]}, "query"),
    ({"id": "a", "group": "no-answer", "tool": "search_memories", "query": "x", "expected": ["m"]},
     "no-answer"),
    ({"id": "a", "group": "superseded", "tool": "search_memories", "query": "x", "expected": ["m"]},
     "superseded"),
])
def test_load_queries_rejects_bad_lines_with_line_number(tmp_path, line, needle):
    path = tmp_path / "q.jsonl"
    good = {"id": "ok", "group": "identifier", "tool": "search_memories", "query": "x", "expected": ["m"]}
    path.write_text(json.dumps(good) + "\n" + json.dumps(line) + "\n")
    with pytest.raises(ValueError) as exc:
        ev.load_queries(path)
    assert "line 2" in str(exc.value)
    assert needle in str(exc.value)


def test_load_queries_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "q.jsonl"
    line = {"id": "dup", "group": "identifier", "tool": "search_memories", "query": "x", "expected": ["m"]}
    path.write_text(json.dumps(line) + "\n\n" + json.dumps(line) + "\n")
    with pytest.raises(ValueError, match="line 3.*dup"):
        ev.load_queries(path)


# --- Metric math ---------------------------------------------------------------


def test_single_hit_at_rank_two():
    q = _queries()["q-ident-1"]
    s = ev.score_query(q, ["s9:1", "s1:10", "s9:2"])
    assert s["recall_at_5"] == 1.0
    assert s["rr"] == 0.5
    assert s["ndcg_at_10"] == pytest.approx(1 / math.log2(3))
    assert s["rank"] == 2


def test_two_expected_one_inside_top_five():
    q = _queries()["q-para-1"]
    s = ev.score_query(q, ["m-a", "x", "y", "z", "w", "m-b"])
    assert s["recall_at_5"] == 0.5
    assert s["rr"] == 1.0
    ideal = 1 + 1 / math.log2(3)
    assert s["ndcg_at_10"] == pytest.approx((1 + 1 / math.log2(7)) / ideal)


def test_miss_scores_zero():
    s = ev.score_query(_queries()["q-buried-1"], ["s9:1", "s9:2"])
    assert (s["recall_at_5"], s["rr"], s["ndcg_at_10"], s["rank"]) == (0.0, 0.0, 0.0, None)


def test_hit_past_rank_ten_counts_only_for_rr():
    ranked = [f"x:{i}" for i in range(11)] + ["s2:40"]
    s = ev.score_query(_queries()["q-buried-1"], ranked)
    assert s["ndcg_at_10"] == 0.0
    assert s["recall_at_5"] == 0.0
    assert s["rr"] == pytest.approx(1 / 12)


def test_no_answer_has_no_recall_and_keeps_top_score():
    s = ev.score_query(_queries()["q-none-1"], ["s9:1"], top_score=0.41)
    assert s["recall_at_5"] is None and s["rr"] is None and s["ndcg_at_10"] is None
    assert s["top_score"] == 0.41


def test_superseded_requires_current_above_old():
    q = _queries()["q-super-1"]
    assert ev.score_query(q, ["m-old", "m-new"])["current_first"] is False
    assert ev.score_query(q, ["m-new", "m-old"])["current_first"] is True
    assert ev.score_query(q, ["m-new"])["current_first"] is True
    assert ev.score_query(q, ["m-old"])["current_first"] is False


def test_summarize_means_per_group_and_overall():
    qs = _queries()
    scored = [
        ev.score_query(qs["q-ident-1"], ["s9:1", "s1:10"]),
        ev.score_query(qs["q-para-1"], ["m-a", "x", "y", "z", "w", "m-b"]),
        ev.score_query(qs["q-none-1"], [], top_score=None),
        ev.score_query(qs["q-super-1"], ["m-new", "m-old"]),
    ]
    summary = ev.summarize(scored)
    assert summary["identifier"]["n"] == 1
    assert summary["identifier"]["mrr"] == 0.5
    assert summary["paraphrase"]["recall_at_5"] == 0.5
    assert summary["no-answer"]["recall_at_5"] is None
    assert summary["superseded"]["current_first"] == 1.0
    # Overall means skip no-answer queries, which have no expected result.
    assert summary["all"]["n"] == 4
    assert summary["all"]["mrr"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)


# --- Tool payloads -------------------------------------------------------------


def test_result_ids_and_top_score_for_turns():
    payload = {"results": [{"session_id": "s1", "turn_index": 10, "distance": 0.25},
                           {"session_id": "s2", "turn_index": 3, "distance": None}]}
    assert ev.result_ids("search_all_sessions", payload) == ["s1:10", "s2:3"]
    assert ev.top_score("search_all_sessions", payload) == pytest.approx(0.75)


def test_result_ids_for_memories_have_no_score():
    payload = {"results": [{"memory_id": "m-1"}, {"memory_id": "m-2"}]}
    assert ev.result_ids("search_memories", payload) == ["m-1", "m-2"]
    assert ev.top_score("search_memories", payload) is None
    assert ev.top_score("search_all_sessions", {"results": []}) is None


# --- Excluded roots ------------------------------------------------------------


@pytest.mark.parametrize("value,needle", [
    ("/x/SessionFlow", "exclude_roots"),
    ([], "exclude_roots"),
    (["relative/path"], "absolute"),
    ([42], "absolute"),
])
def test_load_queries_rejects_bad_exclude_roots(tmp_path, value, needle):
    line = {"id": "a", "group": "cross-project", "tool": "search_all_sessions", "query": "x",
            "args": {"project_root": "*"}, "exclude_roots": value, "expected": ["s:1"]}
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(line) + "\n")
    with pytest.raises(ValueError, match=needle):
        ev.load_queries(path)


class SearchClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return {"results": self.rows[:arguments["n"]]}


def _row(session, turn, root):
    return {"session_id": session, "turn_index": turn, "project_root": root, "distance": 0.2}


def test_excluded_root_and_its_subfolders_are_dropped_before_scoring():
    q = _queries()["q-cross-2"]
    rows = [_row("eval", 1, "/x/SessionFlow"),
            _row("wt", 3, "/x/SessionFlow/.worktrees/a"),
            _row("old", 4, "/x/SessionFlowOld"),
            _row("s5", 2, "/y/HomeNetwork")]
    client = SearchClient(rows)
    [scored] = ev.run_queries(client, [q], "hybrid")
    assert scored["ranked"] == ["old:4", "s5:2"]
    assert scored["rank"] == 2
    assert scored["dropped"] == 2
    assert [a["n"] for _, a in client.calls] == [ev.FETCH_N, ev.FETCH_N * 3]


def test_excluded_rows_leave_at_most_fetch_n_scored():
    q = _queries()["q-cross-2"]
    rows = [_row("eval", i, "/x/SessionFlow") for i in range(5)]
    rows += [_row("other", i, "/y/Other") for i in range(ev.FETCH_N + 5)]
    [scored] = ev.run_queries(SearchClient(rows), [q], "hybrid")
    assert scored["dropped"] == 5
    assert scored["ranked"] == [f"other:{i}" for i in range(ev.FETCH_N)]


def test_full_fill_with_too_few_rows_marks_the_query_short():
    q = _queries()["q-cross-2"]
    rows = [_row("eval", i, "/x/SessionFlow") for i in range(ev.FETCH_N * 3 - 2)]
    rows += [_row("other", i, "/y/Other") for i in range(5)]
    [scored] = ev.run_queries(SearchClient(rows), [q], "hybrid")
    assert scored["ranked"] == ["other:0", "other:1"]
    assert scored["short"] is True
    report = ev.build_report([scored], run={})
    assert "Short after exclusion: q-cross-2" in ev.render_markdown(report)


def test_a_fill_that_runs_out_of_results_is_not_short():
    q = _queries()["q-cross-2"]
    [scored] = ev.run_queries(SearchClient([_row("eval", 1, "/x/SessionFlow"), _row("s", 1, "/y")]),
                              [q], "hybrid")
    assert scored["ranked"] == ["s:1"]
    assert scored["short"] is False


def test_no_excluded_row_keeps_the_normal_search():
    q = _queries()["q-cross-2"]
    client = SearchClient([_row("s5", 2, "/y/Other"), _row("s6", 1, "/y/Other")])
    [scored] = ev.run_queries(client, [q], "hybrid")
    assert [a["n"] for _, a in client.calls] == [ev.FETCH_N]
    assert scored["ranked"] == ["s5:2", "s6:1"]
    assert scored["dropped"] == 0


def test_fill_keeps_the_order_of_the_normal_top_rows():
    class Reordering(SearchClient):
        def call(self, name, arguments):
            self.calls.append((name, arguments))
            if arguments["n"] == ev.FETCH_N:
                return {"results": [_row("eval", 1, "/x/SessionFlow"), _row("b", 1, "/y"),
                                    _row("a", 1, "/y")]}
            return {"results": [_row("a", 1, "/y"), _row("c", 1, "/y"), _row("b", 1, "/y")]}

    [scored] = ev.run_queries(Reordering([]), [_queries()["q-cross-2"]], "hybrid")
    assert scored["ranked"] == ["b:1", "a:1", "c:1"]
    assert scored["dropped"] == 1


@pytest.mark.parametrize("root,row_root,dropped", [
    ("/x/SessionFlow/.", "/x/SessionFlow", 1),
    ("/x/SessionFlow/", "/x/SessionFlow/sub/../a", 1),
    ("/", "/y/Other", 1),
    ("/", "unknown", 0),
    ("/", None, 0),
])
def test_drop_excluded_normalizes_paths(root, row_root, dropped):
    payload = {"results": [_row("s", 1, row_root)]}
    _, count = ev.drop_excluded("search_all_sessions", payload, [root])
    assert count == dropped


def test_top_score_comes_from_the_first_kept_row():
    q = _queries()["q-cross-2"]
    rows = [{**_row("eval", 1, "/x/SessionFlow"), "distance": 0.1}, _row("s5", 2, "/y/Other")]
    [scored] = ev.run_queries(SearchClient(rows), [q], "hybrid")
    assert scored["top_score"] == pytest.approx(0.8)


def test_memory_queries_exclude_by_source_project_root():
    q = {"id": "m", "group": "paraphrase", "tool": "search_memories", "query": "x",
         "args": {"project": "*"}, "exclude_roots": ["/x/SessionFlow"], "expected": ["m-b"]}
    rows = [{"memory_id": "m-a", "source_project_root": "/x/SessionFlow"},
            {"memory_id": "m-b", "source_project_root": "/y/Other"},
            {"memory_id": "m-c", "source_project_root": None}]
    [scored] = ev.run_queries(SearchClient(rows), [q], "hybrid")
    assert scored["ranked"] == ["m-b", "m-c"]
    assert scored["dropped"] == 1


def test_queries_without_exclude_roots_run_unchanged():
    q = _queries()["q-cross-1"]
    client = SearchClient([_row("eval", 1, "/x/SessionFlow"), _row("s4", 1, "/y/Other")])
    [scored] = ev.run_queries(client, [q], "hybrid")
    assert client.calls[0][1]["n"] == ev.FETCH_N
    assert scored["ranked"] == ["eval:1", "s4:1"]
    assert "dropped" not in scored


def test_report_lists_excluded_roots_and_dropped_rows(tmp_path):
    qs = _queries()
    client = SearchClient([_row("eval", 1, "/x/SessionFlow"), _row("s5", 2, "/y/Other")])

    class Health(SearchClient):
        def health(self):
            return {"version": "0", "ranking": {}}

    client = Health(client.rows)
    report = ev.evaluate(client, [qs["q-cross-1"], qs["q-cross-2"]], sort_by="hybrid", env={},
                         queries_file="fixture")
    assert report["run"]["exclude_roots"] == ["/x/SessionFlow"]
    assert report["groups"]["cross-project"]["dropped"] == 1
    md = ev.render_markdown(report)
    assert "- exclude_roots: /x/SessionFlow" in md
    assert "Dropped rows: q-cross-2 1" in md


# --- Ranking settings ----------------------------------------------------------


def test_parse_env_pairs():
    assert ev.parse_env(["SESSIONFLOW_RECENCY_WEIGHT=0.15", "A=b=c"]) == {
        "SESSIONFLOW_RECENCY_WEIGHT": "0.15", "A": "b=c"}
    with pytest.raises(ValueError):
        ev.parse_env(["NOEQUALS"])


def test_check_ranking_reports_mismatch_and_unverified():
    health = {"ranking": {"recency_weight": 0.3, "recency_decay_days": 7}}
    assert ev.check_ranking({"SESSIONFLOW_RECENCY_WEIGHT": "0.3"}, health) == ([], [])
    problems, unverified = ev.check_ranking(
        {"SESSIONFLOW_RECENCY_WEIGHT": "0.15", "SESSIONFLOW_CANDIDATE_POOL": "200"}, health)
    assert len(problems) == 1 and "0.15" in problems[0]
    assert unverified == ["SESSIONFLOW_CANDIDATE_POOL"]
    problems, _ = ev.check_ranking({"SESSIONFLOW_RECENCY_WEIGHT": "0.3"}, {})
    assert problems and "ranking" in problems[0]


def test_check_ranking_verifies_the_temporal_weight():
    health = {"ranking": {"recency_weight": 0.15, "temporal_recency_weight": 0.7}}
    var = "SESSIONFLOW_TEMPORAL_RECENCY_WEIGHT"
    assert ev.check_ranking({var: "0.7"}, health) == ([], [])
    problems, unverified = ev.check_ranking({var: "0.5"}, health)
    assert len(problems) == 1 and "0.5" in problems[0]
    assert unverified == []


# --- Reports -------------------------------------------------------------------


def _report():
    qs = _queries()
    scored = [ev.score_query(qs["q-ident-1"], ["s1:10"]), ev.score_query(qs["q-para-1"], ["m-b"])]
    return ev.build_report(scored, run={"sort_by": "hybrid", "env": {}, "queries_file": "fixture"})


def test_report_file_shape(tmp_path):
    json_path, md_path = ev.write_report(_report(), tmp_path, "2026-10-30-hybrid")
    data = json.loads(json_path.read_text())
    assert set(data) == {"run", "groups", "queries"}
    assert data["run"]["sort_by"] == "hybrid"
    assert data["run"]["query_count"] == 2
    assert {"identifier", "paraphrase", "all"} <= set(data["groups"])
    q = data["queries"][0]
    assert {"id", "group", "tool", "query", "expected", "ranked", "rank",
            "recall_at_5", "rr", "ndcg_at_10"} <= set(q)
    md = md_path.read_text()
    assert "| group | n | recall@5 | MRR | nDCG@10 |" in md
    assert "| identifier | 1 | 1.000 | 1.000 | 1.000 |" in md


def test_comparison_table_has_one_column_per_label():
    md = ev.render_comparison({"w=0": _report(), "w=0.3": _report()}, metric="mrr")
    header = md.splitlines()[0]
    assert "w=0" in header and "w=0.3" in header
    assert any(line.startswith("| identifier |") for line in md.splitlines())


# --- Usage rollup --------------------------------------------------------------


def _at(seconds):
    base = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).isoformat()


def test_usage_rollup_counts_top_three_follow_ups_and_reformulations():
    entries = [
        # Client a: search, open rank 2, then open rank 5.
        {"ts": _at(0), "kind": "search", "tool": "search_memories", "client": "a",
         "ids": ["m1", "m2", "m3", "m4", "m5"]},
        {"ts": _at(5), "kind": "follow_up", "tool": "get_memory", "client": "a", "id": "m2"},
        {"ts": _at(9), "kind": "follow_up", "tool": "get_memory", "client": "a", "id": "m5"},
        # Client b: search, re-search within 60 s with no follow-up, then a late search.
        {"ts": _at(0), "kind": "search", "tool": "search_all_sessions", "client": "b", "ids": ["s:1"]},
        {"ts": _at(30), "kind": "search", "tool": "search_all_sessions", "client": "b", "ids": ["s:2"]},
        {"ts": _at(200), "kind": "search", "tool": "search_all_sessions", "client": "b", "ids": ["s:3"]},
        # Outside the window.
        {"ts": _at(-9 * 86400), "kind": "search", "tool": "search_memories", "client": "c", "ids": []},
    ]
    now = datetime(2026, 10, 30, 13, 0, tzinfo=timezone.utc)
    roll = ev.usage_rollup(entries, now=now, days=7)
    assert roll["searches"] == 4
    assert roll["follow_ups"] == 2
    assert roll["follow_up_top3_share"] == 0.5
    assert roll["reformulated_searches"] == 1
    assert roll["reformulation_share"] == 0.25


def test_usage_rollup_skips_eval_clients():
    eval_client = f"{ev.EVAL_CLIENT_NAME}:1f"
    entries = [
        {"ts": _at(i), "kind": "search", "tool": "search_memories", "client": eval_client, "ids": []}
        for i in range(10)
    ] + [{"ts": _at(20), "kind": "search", "tool": "search_memories", "client": "claude-code:2a",
          "ids": ["m1"]}]
    roll = ev.usage_rollup(entries, now=datetime(2026, 10, 30, 13, 0, tzinfo=timezone.utc), days=7)
    assert roll["searches"] == 1
    assert roll["reformulated_searches"] == 0
    assert roll["eval_calls_skipped"] == 10


def test_usage_rollup_empty_window():
    roll = ev.usage_rollup([], now=datetime.now(timezone.utc), days=7)
    assert roll["searches"] == 0
    assert roll["follow_up_top3_share"] is None
    assert roll["reformulation_share"] is None


# --- Freshness probe -----------------------------------------------------------


class FakeClient:
    def __init__(self, hit_after=3, delete_fails=False, stored_text=None):
        self.calls = []
        self.hit_after = hit_after
        self.delete_fails = delete_fails
        self.searches = 0
        self.stored_text = stored_text

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "add_memory":
            self.stored_text = self.stored_text or arguments["text"]
            return {"memory_id": "canary-1", "outcome": "created", "index": "ok"}
        if name == "search_memories":
            self.searches += 1
            hit = self.hit_after is not None and self.searches >= self.hit_after
            row = {"memory_id": "canary-1", "text": self.stored_text}
            return {"results": [row] if hit else [],
                    "channels": ["fts"] if hit else []}
        if name == "delete_memory":
            if self.delete_fails:
                raise ev.EvalError("delete failed")
            return {"memory_id": "canary-1", "outcome": "deleted"}
        raise AssertionError(name)

    def health(self):
        return {"memory": {"ack_lag": {"milvus": 0}, "index": {
            "host": "minitron", "last_tick_age_s": 1.5, "stale": False}}}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_freshness_probe_reports_time_to_hit_and_deletes():
    client, clock = FakeClient(hit_after=3), FakeClock()
    result = ev.freshness_probe(client, sleep=clock.sleep, clock=clock)
    assert result["time_to_hit_s"] == pytest.approx(2.0)
    assert result["polls"] == 3
    assert result["host"] == "minitron"
    assert result["add_index"] == "ok"
    assert result["health"]["last_tick_age_s"] == 1.5
    assert result["deleted"] is True
    add = client.calls[0][1]
    # One canary per host, so probes on two machines never share a record.
    assert add["topic_key"] == f"{ev.CANARY_TOPIC}:{ev.socket.gethostname()}"
    assert add["project"] == "sessionflow" and add["type"] == "fact"
    token = client.calls[1][1]["query"]
    assert token in add["text"]
    assert client.calls[-1][0] == "delete_memory"


def test_freshness_probe_ignores_a_record_another_probe_rewrote():
    client, clock = FakeClient(hit_after=1, stored_text=ev.canary_text("sfcanary1")), FakeClock()
    result = ev.freshness_probe(client, timeout_s=3, sleep=clock.sleep, clock=clock)
    assert result["time_to_hit_s"] is None


def test_freshness_probes_on_one_host_take_turns(tmp_path):
    import threading

    lock = tmp_path / "freshness.lock"
    order = []

    class SlowClient(FakeClient):
        def call(self, name, arguments):
            if name == "add_memory":
                order.append("add")
            if name == "delete_memory":
                order.append("delete")
            return super().call(name, arguments)

    def probe():
        ev.freshness_probe(SlowClient(hit_after=2), lock_path=lock, poll_s=0.05)

    threads = [threading.Thread(target=probe) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert order == ["add", "delete", "add", "delete"]


@pytest.mark.parametrize("given,want", [
    ("http://127.0.0.1:7102", "http://127.0.0.1:7102/mcp/"),
    ("http://127.0.0.1:7102/", "http://127.0.0.1:7102/mcp/"),
    ("http://127.0.0.1:7102/mcp", "http://127.0.0.1:7102/mcp"),
    ("http://127.0.0.1:7102/mcp/", "http://127.0.0.1:7102/mcp/"),
])
def test_server_url_points_at_the_mcp_endpoint(given, want):
    assert ev.McpClient(given).url == want


def test_freshness_probe_times_out_and_still_deletes():
    client, clock = FakeClient(hit_after=None), FakeClock()
    result = ev.freshness_probe(client, timeout_s=5, sleep=clock.sleep, clock=clock)
    assert result["time_to_hit_s"] is None
    assert result["deleted"] is True
    assert client.calls[-1][0] == "delete_memory"


def test_freshness_probe_reports_failed_delete():
    client, clock = FakeClient(delete_fails=True), FakeClock()
    result = ev.freshness_probe(client, sleep=clock.sleep, clock=clock)
    assert result["deleted"] is False
    assert "delete failed" in result["delete_error"]


def test_canary_text_passes_the_memory_secret_check():
    for _ in range(20):
        text = ev.canary_text(ev.make_token())
        _, spans = secret_redaction.scan_spans(text, mode="report")
        assert spans == []


def test_failed_add_deletes_only_an_exact_canary():
    class AddFails(FakeClient):
        def call(self, name, arguments):
            if name == "add_memory":
                self.calls.append((name, arguments))
                self.stored_text = arguments["text"]
                raise ev.EvalError("timeout after commit")
            if name == "search_memories":
                self.calls.append((name, arguments))
                return {"results": [{"memory_id": "other", "text": "unrelated"},
                                    {"memory_id": "canary-1", "text": self.stored_text}]}
            return super().call(name, arguments)

    client = AddFails()
    result = ev.freshness_probe(client)
    assert result["error"] == "timeout after commit"
    assert result["deleted"] is True
    deletes = [a for n, a in client.calls if n == "delete_memory"]
    assert [d["memory_id"] for d in deletes] == ["canary-1"]


def test_restore_server_runs_every_step_and_reports_failures(monkeypatch):
    import subprocess

    steps = []

    def launchctl(*args):
        steps.append(args)
        raise subprocess.CalledProcessError(1, "launchctl")

    monkeypatch.setattr(ev, "_launchctl", launchctl)
    monkeypatch.setattr(ev, "_restart_server", lambda label: steps.append(("restart", label)))
    monkeypatch.setattr(ev, "wait_ready", lambda client, env: steps.append(("ready", env)))
    errors = ev.restore_server("SESSIONFLOW_RECENCY_WEIGHT", "0.2", "lbl", "http://x/mcp/")
    assert steps == [("setenv", "SESSIONFLOW_RECENCY_WEIGHT", "0.2"), ("restart", "lbl"),
                     ("ready", {"SESSIONFLOW_RECENCY_WEIGHT": "0.2"})]
    assert len(errors) == 1


def test_restore_server_unsets_a_value_that_was_unset(monkeypatch):
    steps = []
    monkeypatch.setattr(ev, "_launchctl", lambda *args: steps.append(args) or "")
    monkeypatch.setattr(ev, "_restart_server", lambda label: None)
    monkeypatch.setattr(ev, "wait_ready", lambda client, env: steps.append(env))
    assert ev.restore_server("SESSIONFLOW_RECENCY_WEIGHT", None, "lbl", "http://x/mcp/") == []
    assert steps == [("unsetenv", "SESSIONFLOW_RECENCY_WEIGHT"), {}]


def test_eval_agent_plist_escapes_paths_and_keeps_the_url(tmp_path):
    import plistlib
    import subprocess

    script = Path(__file__).resolve().parent.parent / "sessionflow-server.sh"
    plist = tmp_path / "eval.plist"
    shell = (
        'eval "$(sed -n "/^xml_escape()/,/^}/p;/^write_eval_agent_plist()/,/^}/p" "$0")"; '
        f'LAUNCH_AGENT_DIR="{tmp_path}"; EVAL_AGENT_PLIST="{plist}"; EVAL_AGENT_LABEL=l; '
        'PYTHON=/p/python; SCRIPT_DIR=/repo; EVAL_HOUR=3; EVAL_MINUTE=30; '
        'EVAL_AGENT_STDOUT=/o; EVAL_AGENT_STDERR=/e; '
        'SESSIONFLOW_EVAL_QUERIES="/a & b/q.jsonl"; SESSIONFLOW_EVAL_OUT="/a <b>/eval"; '
        'SESSIONFLOW_PORT=7200; write_eval_agent_plist'
    )
    subprocess.run(["bash", "-c", shell, str(script)], check=True,
                   env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    args = plistlib.loads(plist.read_bytes())["ProgramArguments"]
    assert args[2:4] == ["--url", "http://127.0.0.1:7200"]
    assert "/a & b/q.jsonl" in args and "/a <b>/eval" in args
