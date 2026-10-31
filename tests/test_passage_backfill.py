"""cleanup.py passages: re-parse turns cut at the old limit, then add passages to long turns."""

from __future__ import annotations

import contextlib
import types

import pytest

import passage_backfill
import rag_engine
import reembed
import transcript_parser
from passages import PASSAGE_CHUNK_TYPE, PASSAGE_TOKENS

REAL_REPLACE_PASSAGES = rag_engine.replace_passages
REAL_WRITE_PASSAGE_FTS = rag_engine.write_passage_fts
OLD_CUT = ("word " * transcript_parser.MAX_TURN_CHARS)[:transcript_parser.MAX_TURN_CHARS] \
    + transcript_parser.CUT_MARK


def _words(n, user="why is the tail missing"):
    return f"User: {user}\n\nAssistant: " + " ".join(f"w{i}" for i in range(n))


def _word_offsets(text):
    spans, pos = [], 0
    for word in text.split():
        start = text.index(word, pos)
        spans.append((start, start + len(word)))
        pos = start + len(word)
    return spans


def _row(pk, doc_id, document, *, session="s", turn_index=1, provider="claude_code_cli",
         chunk_type="turn"):
    return {"id": pk, "doc_id": doc_id, "document": document, "session_id": session,
            "logical_session_id": session, "turn_index": turn_index, "provider": provider,
            "chunk_type": chunk_type, "project_root": "/p/Proj", "issue_ids": ",SESF-1,",
            "timestamp": "2026-10-01T00:00:00Z", "git_branch": "main", "source_kind": "k",
            "source_class": "c", "source_id": "src", "source_path": "/t.jsonl",
            "transcript_file": "t.jsonl", "vector": [0.0]}


@pytest.fixture(autouse=True)
def no_live_index(monkeypatch):
    """Fail any read or write that would reach the real index; ``world`` stubs what it needs."""
    def _refuse(*args, **kwargs):
        raise AssertionError("test reached the live index")

    for name in ("_query_batches", "add_turns", "get_rows_by_ids", "replace_passages",
                 "write_passage_fts",
                 "delete_by_doc_id", "milvus_client", "embed_texts"):
        monkeypatch.setattr(rag_engine, name, _refuse)
    monkeypatch.setattr(reembed, "_embed_documents", _refuse)
    monkeypatch.setattr(passage_backfill, "_stored_doc_ids", _refuse)
    monkeypatch.setattr(passage_backfill, "_sources", _refuse)


class _Budget:
    batch_size = 2

    def __init__(self):
        self.deny = False


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    w = types.SimpleNamespace(
        rows={}, parsed={}, added=[], replaced=[], embedded=[], budget=_Budget(),
        stored=set(), errors={}, deny_after=None,
    )

    def _batches(fields, filter_expr=None, db_path=None, **kw):
        yield [dict(r) for r in w.rows.values()]

    def _add_turns(turns, db_path=None):
        w.added.append(list(turns))
        for t in turns:
            w.stored.add(t["doc_id"])
            for r in [r for r in w.rows.values()
                      if r["session_id"] == t["session_id"]
                      and r["turn_index"] == t["turn_index"] and r["chunk_type"] == "turn"]:
                del w.rows[r["id"]]
            pk = 1000 + len(w.rows) + len(w.added)
            w.rows[pk] = _row(pk, t["doc_id"], t["text"], session=t["session_id"],
                              turn_index=t["turn_index"], provider=t.get("provider", "claude_code_cli"))
        return len(turns)

    def _get_rows(ids, db_path=None):
        return [dict(w.rows[i]) for i in ids if i in w.rows]

    def _replace(parent_doc_id, built, vectors, db_path=None):
        w.replaced.append((parent_doc_id, [p["doc_id"] for p in built], len(vectors)))
        for i, p in enumerate(built):
            pk = 5000 + len(w.rows)
            w.rows[pk] = _row(pk, p["doc_id"], p["text"], session=p["session_id"],
                              turn_index=p["turn_index"], chunk_type=PASSAGE_CHUNK_TYPE)
        return True

    def _embed(budget, texts, titles=None):
        if w.deny_after is not None and len(w.embedded) >= w.deny_after:
            raise reembed._BudgetDenied("cap")
        w.embedded.append((list(texts), list(titles or [])))
        return [[0.0] for _ in texts]

    def _sources(keys):
        return {key: [types.SimpleNamespace(key=key)] for key in keys if key in w.parsed}

    def _parse(provider, source):
        key = source.key
        return types.SimpleNamespace(turns=w.parsed[key], errors=w.errors.get(key, []))

    monkeypatch.setattr(rag_engine, "_query_batches", _batches)
    monkeypatch.setattr(rag_engine, "add_turns", _add_turns)
    monkeypatch.setattr(rag_engine, "get_rows_by_ids", _get_rows)
    monkeypatch.setattr(rag_engine, "replace_passages", _replace)
    monkeypatch.setattr(rag_engine, "_token_offsets", _word_offsets)
    monkeypatch.setattr(reembed, "_embed_documents", _embed)
    monkeypatch.setattr(reembed, "_db_path", lambda: "/tmp/sesf-test-backfill/milvus.db")
    monkeypatch.setattr(passage_backfill, "_sources", _sources)
    monkeypatch.setattr(passage_backfill, "_parse", _parse)
    monkeypatch.setattr(passage_backfill, "_stored_doc_ids",
                        lambda doc_ids, db_path=None: set(doc_ids) & w.stored)
    monkeypatch.setattr(passage_backfill, "get_embedding_budget", lambda: w.budget)
    return w


def _parsed_turn(doc_id, text, turn_index=1, session="s", provider="claude_code_cli"):
    return {"doc_id": doc_id, "text": text, "session_id": session, "logical_session_id": session,
            "turn_index": turn_index, "chunk_type": "turn", "provider": provider,
            transcript_parser.REPLACE_SAME_TURN: True}


# --- selection ----------------------------------------------------------------


def test_old_cut_is_recognized_and_new_cuts_are_not():
    assert passage_backfill.is_old_cut(OLD_CUT)
    assert not passage_backfill.is_old_cut("x" * 9000 + transcript_parser.CUT_MARK)
    assert not passage_backfill.is_old_cut("short reply")


def test_plan_counts_cut_sessions_and_turns_without_passages(world):
    world.rows = {
        1: _row(1, "s::cut", OLD_CUT, turn_index=1),
        2: _row(2, "s::long", _words(1000), turn_index=2),
        3: _row(3, "s::done", _words(1000), turn_index=3),
        4: _row(4, "s::done::p0", "w1 w2", turn_index=3, chunk_type=PASSAGE_CHUNK_TYPE),
        5: _row(5, "s::short", "User: q\n\nAssistant: a", turn_index=4),
        6: _row(6, "c::cut", OLD_CUT, session="c", provider="codex"),
        7: _row(7, "a::cut", OLD_CUT, session="a", provider="antigravity_cli"),
    }
    report = passage_backfill.plan()
    assert report.status == "dry-run"
    assert report.cut_rows == 2  # antigravity has no turn re-parse
    assert report.sessions == 2
    assert report.missing == 4  # both cut turns, s::long and a::cut
    assert world.added == [] and world.replaced == []


def test_run_refuses_without_confirmation(world):
    assert passage_backfill.run(confirmed=False).status == "refused"
    assert world.added == []


# --- uncut phase --------------------------------------------------------------


def test_cut_turns_are_reparsed_and_replaced(world):
    world.rows = {1: _row(1, "s::cut", OLD_CUT, turn_index=1),
                  2: _row(2, "s::other", "User: q\n\nAssistant: fine", turn_index=2)}
    full = _words(3000)
    world.parsed = {("claude_code_cli", "s"): [
        _parsed_turn("s::full", full, turn_index=1),
        _parsed_turn("s::other", "User: q\n\nAssistant: fine", turn_index=2),
    ]}
    report = passage_backfill.run(confirmed=True)

    assert report.status == "done"
    assert [[t["doc_id"] for t in batch] for batch in world.added] == [["s::full"]]
    assert report.uncut == 1
    assert all(r["doc_id"] != "s::cut" for r in world.rows.values())


def test_deferred_reparse_pauses_and_resumes(world):
    world.rows = {1: _row(1, "s::cut", OLD_CUT, turn_index=1)}
    world.parsed = {("claude_code_cli", "s"): [_parsed_turn("s::full", _words(30), turn_index=1)]}
    real_add = rag_engine.add_turns
    rag_engine.add_turns = lambda turns, db_path=None: 0  # budget deferred everything

    report = passage_backfill.run(confirmed=True)
    assert report.status == "paused"
    assert report.cursor == 0

    rag_engine.add_turns = real_add
    report = passage_backfill.run(confirmed=True)
    assert report.status == "done"
    assert report.uncut == 1


def test_gone_and_unreadable_sessions_are_counted_and_skipped(world):
    world.rows = {1: _row(1, "g::cut", OLD_CUT, session="g"),
                  2: _row(2, "u::cut", OLD_CUT, session="u")}
    world.parsed = {("claude_code_cli", "u"): [_parsed_turn("u::full", _words(30), session="u")]}
    world.errors = {("claude_code_cli", "u"): ["bad line"]}
    report = passage_backfill.run(confirmed=True)
    assert report.gone == 1
    assert report.unreadable == 1
    assert world.added == []
    # Both cut rows stay and get passages from their stored text.
    assert {d for d, _, _ in world.replaced} == {"g::cut", "u::cut"}


# --- passages phase -----------------------------------------------------------


def test_long_turns_without_passages_get_them_with_titles(world):
    world.rows = {2: _row(2, "s::long", _words(1000), turn_index=2),
                  3: _row(3, "s::short", "User: q\n\nAssistant: " + "a " * (PASSAGE_TOKENS // 2),
                          turn_index=3)}
    report = passage_backfill.run(confirmed=True)

    assert report.status == "done"
    assert [d for d, _, _ in world.replaced] == ["s::long"]
    parent, ids, n_vectors = world.replaced[0]
    assert ids == [f"s::long::p{i}" for i in range(len(ids))] and n_vectors == len(ids)
    _, titles = world.embedded[0]
    assert set(titles) == {"Proj · why is the tail missing"}
    assert report.added == len(ids)
    assert report.short == 1


def test_passages_phase_pauses_on_budget_and_resumes(world):
    world.rows = {i: _row(i, f"s::l{i}", _words(1000), turn_index=i) for i in range(1, 4)}
    world.deny_after = 2  # the first turn embeds in two budget-sized calls
    report = passage_backfill.run(confirmed=True)
    assert report.status == "paused"
    assert len(world.replaced) == 1

    world.deny_after = None
    report = passage_backfill.run(confirmed=True)
    assert report.status == "done"
    assert sorted(d for d, _, _ in world.replaced) == ["s::l1", "s::l2", "s::l3"]


def test_max_turns_pauses_the_chunk(world):
    world.rows = {i: _row(i, f"s::l{i}", _words(1000), turn_index=i) for i in range(1, 4)}
    report = passage_backfill.run(confirmed=True, max_turns=1)
    assert report.status == "paused"
    assert len(world.replaced) == 1


def test_finished_run_reports_done_and_restart_rescans(world):
    world.rows = {2: _row(2, "s::long", _words(1000), turn_index=2)}
    assert passage_backfill.run(confirmed=True).status == "done"
    assert passage_backfill.run(confirmed=True).status == "done"
    assert len(world.replaced) == 1
    report = passage_backfill.run(confirmed=True, restart=True)
    assert report.status == "done"
    assert len(world.replaced) == 1  # the rescan finds the passages


def test_verify_reports_what_is_left(world):
    world.rows = {1: _row(1, "s::cut", OLD_CUT),
                  2: _row(2, "s::long", _words(1000), turn_index=2)}
    result = passage_backfill.verify()
    assert result.missing == 2
    assert result.cut_rows == 1


# --- rag_engine.replace_passages ----------------------------------------------


def test_replace_passages_rewrites_milvus_and_fts(monkeypatch):
    calls = []

    class _Client:
        def query(self, collection_name, filter, output_fields=None, limit=None):
            calls.append(("query", filter))
            return [{"id": 9, "doc_id": "s::p::p0"}, {"id": 8, "doc_id": "s::other::p0"}]

        def delete(self, collection_name, filter):
            calls.append(("delete", filter))

        def insert(self, collection_name, data):
            calls.append(("insert", [d["doc_id"] for d in data], data[0]["chunk_type"],
                          data[0]["issue_ids"]))

    @contextlib.contextmanager
    def _client(db_path=None):
        yield _Client()

    fts = []
    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    monkeypatch.setattr(rag_engine, "write_passage_fts", REAL_WRITE_PASSAGE_FTS)
    monkeypatch.setattr(rag_engine, "_fts", types.SimpleNamespace(
        connection=lambda db_path: object(),
        delete_where=lambda conn, where, params: fts.append(("delete_where", params)),
        insert=lambda conn, records: fts.append(("insert", [r["doc_id"] for r in records])),
        close_ephemeral=lambda conn: None))

    built = [{"doc_id": f"s::p::p{i}", "text": f"t{i}", "session_id": "s", "turn_index": 4,
              "chunk_type": PASSAGE_CHUNK_TYPE, "_issue_ids": ",SESF-1,"} for i in range(2)]
    ok = REAL_REPLACE_PASSAGES("s::p", built, [[0.0] * rag_engine._EMBED_DIM] * 2,
                                     db_path="/tmp/sesf-test-backfill/milvus.db")

    assert ok
    assert ("delete", "id in [9]") in calls
    assert ("insert", ["s::p::p0", "s::p::p1"], PASSAGE_CHUNK_TYPE, ",SESF-1,") in calls
    assert fts == [("delete_where", ("s::p::p", "s::p::q")), ("insert", ["s::p::p0", "s::p::p1"])]


def test_replace_passages_reports_an_fts_failure(monkeypatch):
    class _Client:
        def query(self, **kw):
            return []

        def insert(self, collection_name, data):
            pass

    @contextlib.contextmanager
    def _client(db_path=None):
        yield _Client()

    def _boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    monkeypatch.setattr(rag_engine, "write_passage_fts", REAL_WRITE_PASSAGE_FTS)
    monkeypatch.setattr(rag_engine, "_fts", types.SimpleNamespace(
        connection=lambda db_path: object(), delete_where=_boom,
        close_ephemeral=lambda conn: None))
    built = [{"doc_id": "s::p::p0", "text": "t", "session_id": "s", "turn_index": 4,
              "chunk_type": PASSAGE_CHUNK_TYPE, "_issue_ids": ""}]
    assert REAL_REPLACE_PASSAGES("s::p", built, [[0.0] * rag_engine._EMBED_DIM],
                                       db_path="/tmp/sesf-test-backfill/milvus.db") is False


# --- CLI ----------------------------------------------------------------------


def test_cli_registers_the_passages_command(monkeypatch, capsys):
    import cleanup

    args = cleanup.build_parser().parse_args(["passages", "run", "--yes", "--max-turns", "5"])
    assert args.action == "run" and args.yes and args.max_turns == 5

    monkeypatch.setattr("sys.argv", ["cleanup.py", "passages", "run"])
    with pytest.raises(SystemExit) as exit_info:
        cleanup.main()
    assert exit_info.value.code == 1
    assert "Refusing to run" in capsys.readouterr().err


def test_a_turn_with_many_passages_embeds_in_budget_batches(world):
    world.rows = {2: _row(2, "s::long", _words(3000), turn_index=2)}
    report = passage_backfill.run(confirmed=True)

    assert report.status == "done"
    assert len(world.embedded) > 1
    assert all(len(texts) <= world.budget.batch_size for texts, _ in world.embedded)
    (parent, ids, n_vectors), = world.replaced
    assert n_vectors == len(ids) == sum(len(texts) for texts, _ in world.embedded)


def test_reparsed_turn_that_ingest_drops_is_not_a_deferral(world):
    world.rows = {1: _row(1, "s::cut", OLD_CUT, turn_index=1)}
    noise = "<system-reminder>\n" + "harness noise " * 50 + "\n</system-reminder>\nok"
    world.parsed = {("claude_code_cli", "s"): [_parsed_turn("s::noise", noise, turn_index=1)]}
    rag_engine.add_turns = lambda turns, db_path=None: 0  # ingest drops the stripped turn

    report = passage_backfill.run(confirmed=True)

    assert report.status == "done"
    assert report.discarded == 1
    assert report.uncut == 0
    assert [d for d, _, _ in world.replaced] == ["s::cut"]


def test_would_store_mirrors_the_boilerplate_drop():
    noise = "<system-reminder>\n" + "harness noise " * 50 + "\n</system-reminder>\nok"
    assert not rag_engine.would_store({"text": noise})
    assert rag_engine.would_store({"text": "User: a real question\n\nAssistant: a reply"})


def test_failed_passage_fts_write_is_retried_before_done(world, monkeypatch):
    world.rows = {2: _row(2, "s::long", _words(1000), turn_index=2)}
    real_replace = rag_engine.replace_passages

    def _replace_fts_fails(parent_doc_id, built, vectors, db_path=None):
        real_replace(parent_doc_id, built, vectors, db_path)
        return False

    retried = []
    monkeypatch.setattr(rag_engine, "replace_passages", _replace_fts_fails)
    monkeypatch.setattr(rag_engine, "write_passage_fts",
                        lambda parent_doc_id, built, db_path=None: retried.append(
                            (parent_doc_id, [p["doc_id"] for p in built])) or False)

    report = passage_backfill.run(confirmed=True)
    assert report.status == "paused"
    assert report.fts_pending == 1

    monkeypatch.setattr(rag_engine, "write_passage_fts",
                        lambda parent_doc_id, built, db_path=None: retried.append(
                            (parent_doc_id, [p["doc_id"] for p in built])) or True)
    report = passage_backfill.run(confirmed=True)
    assert report.status == "done"
    assert report.fts_pending == 0
    assert len(world.embedded) == 2  # the retry rebuilt the text without embedding
    assert {d for d, _ in retried} == {"s::long"}
    assert retried[-1][1] == [f"s::long::p{i}" for i in range(len(retried[-1][1]))]
