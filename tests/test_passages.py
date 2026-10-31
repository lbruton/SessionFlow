"""Passages: long turns get token-bounded passage rows beside the full parent row."""

from __future__ import annotations

import contextlib
import types

import pytest

import passages
import rag_engine
import tools
import transcript_parser
from passages import PASSAGE_CHUNK_TYPE, PASSAGE_OVERLAP, PASSAGE_TOKENS

DB = "/tmp/sesf-test-passages/milvus.db"


def _word_offsets(text):
    """Whitespace tokenizer: one token per word, as (start, end) char spans."""
    spans, pos = [], 0
    for word in text.split():
        start = text.index(word, pos)
        spans.append((start, start + len(word)))
        pos = start + len(word)
    return spans


def _long_text(words, user="how do passages work"):
    body = " ".join(f"w{i}" for i in range(words))
    return f"User: {user}\n\nAssistant: {body}"


# --- splitting ---------------------------------------------------------------


def test_short_turn_has_no_passages():
    text = _long_text(PASSAGE_TOKENS - 10)
    assert passages.passage_spans(_word_offsets(text)) == []


def test_long_turn_is_split_into_overlapping_windows():
    text = _long_text(1000)
    offsets = _word_offsets(text)
    spans = passages.passage_spans(offsets)

    step = PASSAGE_TOKENS - PASSAGE_OVERLAP
    assert spans[0] == (offsets[0][0], offsets[PASSAGE_TOKENS - 1][1])
    assert spans[1][0] == offsets[step][0]
    assert spans[-1][1] == offsets[-1][1]
    for start, end in spans:
        assert len(_word_offsets(text[start:end])) <= PASSAGE_TOKENS
    # every token is inside some passage
    covered = set()
    for start, end in spans:
        covered.update(i for i, (s, e) in enumerate(offsets) if s >= start and e <= end)
    assert covered == set(range(len(offsets)))


def test_last_window_is_not_a_sliver():
    offsets = _word_offsets(_long_text(PASSAGE_TOKENS + 5))
    spans = passages.passage_spans(offsets)
    assert len(spans) == 2
    assert spans[-1] == (offsets[-PASSAGE_TOKENS][0], offsets[-1][1])


def test_special_token_spans_are_ignored():
    offsets = [(0, 0)] + _word_offsets(_long_text(1000)) + [(0, 0)]
    spans = passages.passage_spans(offsets)
    assert spans[0][1] > 0
    assert all(end > start for start, end in spans)


def test_passage_doc_ids_round_trip():
    doc_id = passages.passage_doc_id("s::abc123", 3)
    assert doc_id == "s::abc123::p3"
    assert passages.passage_parent(doc_id) == "s::abc123"
    assert passages.passage_index(doc_id) == 3
    assert passages.passage_parent("s::abc123") is None
    assert passages.passage_index("s::abc123") is None


def test_title_names_the_project_and_the_user_question():
    text = "User: " + "why does the sliding window drift " * 10 + "\n\nAssistant: because"
    title = passages.passage_title("/Volumes/DATA/GitHub/SessionFlow", text)
    project, question = title.split(" · ")
    assert project == "SessionFlow"
    assert question.startswith("why does the sliding window drift")
    assert len(question) <= passages.TITLE_CHARS


def test_title_falls_back_to_none():
    assert passages.passage_title("", "Assistant: only a reply") == "none"
    assert passages.passage_title("unknown", "User: q") == "q"
    assert passages.passage_title("/", "User: q") == "q"


def test_title_collapses_whitespace_and_drops_the_separator():
    title = passages.passage_title("", "User: line one\n  line | two\n\nAssistant: x")
    assert title == "line one line / two"


def test_build_passages_copies_parent_metadata():
    text = _long_text(1000)
    parent = {
        "doc_id": "s::abc", "text": text, "session_id": "s", "turn_index": 42,
        "timestamp": "2026-10-31T10:00:00Z", "git_branch": "main", "chunk_type": "turn",
        "project_root": "/p/Proj", "provider": "claude_code_cli",
        transcript_parser.REPLACE_SAME_TURN: True,
    }
    built = passages.build_passages(parent, _word_offsets(text))

    assert [p["doc_id"] for p in built] == [f"s::abc::p{i}" for i in range(len(built))]
    for p in built:
        assert p["chunk_type"] == PASSAGE_CHUNK_TYPE
        assert p["session_id"] == "s" and p["turn_index"] == 42
        assert p["timestamp"] == parent["timestamp"] and p["provider"] == "claude_code_cli"
        assert p["title"] == "Proj · how do passages work"
        assert transcript_parser.REPLACE_SAME_TURN not in p
        assert p["text"] in text
    assert "text" in parent and parent["chunk_type"] == "turn"


# --- stored turn cap -----------------------------------------------------------


def test_turn_text_keeps_long_turns_whole():
    text = "x" * 20_000
    assert transcript_parser.truncate_turn_text(text) == text


def test_turn_text_cut_fits_the_milvus_byte_cap():
    text = "é" * 40_000  # two bytes each
    cut = transcript_parser.truncate_turn_text(text)
    assert cut.endswith("[truncated]")
    assert len(cut.encode()) <= transcript_parser.MAX_TURN_BYTES


# --- embedding prefix ---------------------------------------------------------


def test_document_prefix_carries_the_title():
    assert rag_engine._document_prefix(None) == rag_engine._DOCUMENT_PREFIX
    if "title: none" in rag_engine._DOCUMENT_PREFIX:
        assert rag_engine._document_prefix("Proj · q") == "title: Proj · q | text: "


def test_embed_texts_passes_titles_to_the_mlx_thread(monkeypatch):
    seen = {}

    def _local(texts, is_query=False, titles=None):
        seen["titles"] = titles
        return [[0.0] for _ in texts]

    class _Executor:
        def submit(self, fn, *args):
            fut = types.SimpleNamespace(result=lambda: fn(*args))
            return fut

    monkeypatch.setattr(rag_engine, "_embed_texts_local", _local)
    monkeypatch.setattr(rag_engine, "_embed_executor", _Executor())
    rag_engine.embed_texts(["a", "b"], is_query=False, titles=["t", None])
    assert seen["titles"] == ["t", None]


# --- add_turns ----------------------------------------------------------------


@pytest.fixture
def index(monkeypatch):
    """In-memory Milvus and FTS stand-ins plus a whitespace tokenizer."""
    rows: dict[str, dict] = {}
    fts_rows: dict[str, dict] = {}
    embedded: list[tuple[list, list]] = []
    budget = {"limit": None}

    class _Decision:
        allowed = True
        retry_after_seconds = 0
        reason = ""

    class _Budget:
        def split_batches(self, turns):
            return [[t] for t in turns]

        def before_batch(self, **kw):
            d = _Decision()
            if budget["limit"] is not None and len(embedded) >= budget["limit"]:
                d = types.SimpleNamespace(allowed=False, retry_after_seconds=0, reason="cap")
            return d

        def after_batch(self, *a, **k):
            pass

    class _Client:
        def query(self, collection_name, filter, output_fields=None, limit=None):
            if filter.startswith("doc_id == "):
                doc_id = filter.split('"')[1]
                return [{"doc_id": doc_id}] if doc_id in rows else []
            return []

        def insert(self, collection_name, data):
            for d in data:
                rows[d["doc_id"]] = d

        def has_collection(self, name):
            return True

    @contextlib.contextmanager
    def _client(db_path=None):
        yield _Client()

    def _embed(texts, is_query=False, titles=None):
        embedded.append((list(texts), list(titles) if titles else [None] * len(texts)))
        return [[0.0] * rag_engine._EMBED_DIM for _ in texts]

    def _insert(conn, records):
        for r in records:
            fts_rows[r["doc_id"]] = r

    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    monkeypatch.setattr(rag_engine, "embed_texts", _embed)
    monkeypatch.setattr(rag_engine, "get_embedding_budget", lambda: _Budget())
    monkeypatch.setattr(rag_engine, "_token_offsets", _word_offsets)
    monkeypatch.setattr(rag_engine, "delete_by_doc_id",
                        lambda doc_id, db_path=None: rag_engine.DeleteResult(1, True))
    monkeypatch.setattr(rag_engine, "_fts", types.SimpleNamespace(
        connection=lambda db_path: object(), insert=_insert,
        close_ephemeral=lambda conn: None))
    return types.SimpleNamespace(rows=rows, fts=fts_rows, embedded=embedded, budget=budget)


def _turn(doc_id, text, turn_index=7, session="s"):
    return {"doc_id": doc_id, "text": text, "session_id": session, "turn_index": turn_index,
            "chunk_type": "turn", "project_root": "/p/Proj",
            transcript_parser.REPLACE_SAME_TURN: True}


def test_long_turn_stores_the_parent_and_its_passages(index):
    text = _long_text(1000)
    stored = rag_engine.add_turns([_turn("s::long", text)], db_path=DB)

    passage_ids = sorted(d for d in index.rows if d != "s::long")
    assert passage_ids and all(d.startswith("s::long::p") for d in passage_ids)
    assert stored == 1 + len(passage_ids)
    assert index.rows["s::long"]["document"] == text
    assert index.rows["s::long"]["chunk_type"] == "turn"
    for d in passage_ids:
        row = index.rows[d]
        assert row["chunk_type"] == PASSAGE_CHUNK_TYPE
        assert row["turn_index"] == 7 and row["session_id"] == "s"
        assert row["issue_ids"] == index.rows["s::long"]["issue_ids"]
    assert set(index.fts) == set(index.rows)


def test_passages_embed_with_a_title_and_the_parent_without(index):
    rag_engine.add_turns([_turn("s::long", _long_text(1000))], db_path=DB)

    titles = [title for _, batch_titles in index.embedded for title in batch_titles]
    assert titles[0] is None
    assert titles[1:] and all(t == "Proj · how do passages work" for t in titles[1:])


def test_short_turn_stores_one_row(index):
    rag_engine.add_turns([_turn("s::short", "User: q\n\nAssistant: a short reply")], db_path=DB)
    assert set(index.rows) == {"s::short"}


def test_a_deferred_batch_drops_the_whole_turn(index):
    index.budget["limit"] = 2  # parent and one passage, then the budget refuses
    rag_engine.add_turns([
        _turn("s::short", "User: q\n\nAssistant: a short reply", turn_index=1),
        _turn("s::long", _long_text(1000), turn_index=7),
    ], db_path=DB)
    assert set(index.rows) == {"s::short"}


def test_passages_come_from_the_stripped_text(index):
    reminder = "<system-reminder>\nsecret harness chatter\n</system-reminder>\n"
    text = "User: how do passages work\n" + reminder + "\n\nAssistant: " + " ".join(
        f"w{i}" for i in range(1000))
    rag_engine.add_turns([_turn("s::long", text)], db_path=DB)
    for doc_id, row in index.rows.items():
        assert "harness chatter" not in row["document"], doc_id


def test_tokenizer_failure_stores_the_parent_alone(index, monkeypatch):
    def _boom(text):
        raise RuntimeError("model not cached")
    monkeypatch.setattr(rag_engine, "_token_offsets", _boom)
    rag_engine.add_turns([_turn("s::long", _long_text(1000))], db_path=DB)
    assert set(index.rows) == {"s::long"}


# --- delete cascade -----------------------------------------------------------


def test_deleting_a_parent_deletes_its_passages(monkeypatch):
    parent = "s::abc"
    rows = {
        rag_engine._pk_from_doc_id(d): {"doc_id": d, "session_id": "s", "turn_index": 7,
                                        "chunk_type": c}
        for d, c in [(parent, "turn"), (f"{parent}::p0", PASSAGE_CHUNK_TYPE),
                     (f"{parent}::p1", PASSAGE_CHUNK_TYPE), ("s::other::p0", PASSAGE_CHUNK_TYPE)]
    }
    fts_deletes = []

    class _Client:
        def has_collection(self, name):
            return True

        def query(self, collection_name, filter, output_fields=None, limit=None):
            if filter.startswith("id == "):
                pk = int(filter.split("== ")[1])
                return [{"id": pk, **rows[pk]}] if pk in rows else []
            if filter.startswith("id in "):
                return []
            assert 'chunk_type == "turn_passage"' in filter
            return [{"id": pk, **r} for pk, r in rows.items()
                    if r["chunk_type"] == PASSAGE_CHUNK_TYPE and r["session_id"] == "s"]

        def delete(self, collection_name, filter):
            if filter.startswith("id == "):
                rows.pop(int(filter.split("== ")[1]), None)
            else:
                for pk in [int(x) for x in filter.split("[")[1].rstrip("]").split(",")]:
                    rows.pop(pk, None)

    @contextlib.contextmanager
    def _client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    monkeypatch.setattr(rag_engine, "_fts", types.SimpleNamespace(
        connection=lambda db_path: object(),
        delete=lambda conn, col, value: fts_deletes.append(("eq", value)),
        delete_where=lambda conn, where, params: fts_deletes.append(("where", params)),
        close_ephemeral=lambda conn: None))

    result = rag_engine.delete_by_doc_id(parent, db_path=DB)

    assert result == rag_engine.DeleteResult(1, True)
    assert [r["doc_id"] for r in rows.values()] == ["s::other::p0"]
    assert ("eq", parent) in fts_deletes
    assert ("where", (f"{parent}::p", f"{parent}::q")) in fts_deletes


# --- search collapse ----------------------------------------------------------


def _row(doc_id, chunk_type="turn", session="s", turn_index=7, content="c", score=0.0):
    return {"doc_id": doc_id, "chunk_type": chunk_type, "session_id": session,
            "turn_index": turn_index, "content": content, "_rrf_score": score}


def test_collapse_keeps_the_best_row_per_turn():
    merged = [
        _row("s::a::p2", PASSAGE_CHUNK_TYPE, content="the matching passage", score=0.9),
        _row("s::a", content="the whole turn", score=0.8),
        _row("s::a::p0", PASSAGE_CHUNK_TYPE, score=0.7),
        _row("s::b", turn_index=9, score=0.6),
        _row("t::a", session="t", score=0.5),
        _row("s::sum", chunk_type="summary", score=0.4),
    ]
    out = rag_engine._collapse_passages(merged)
    assert [r["doc_id"] for r in out] == ["s::a::p2", "s::b", "t::a", "s::sum"]
    assert out[0]["content"] == "the matching passage"
    assert out[0]["passage_index"] == 2


def test_collapse_cuts_a_long_parent_for_display():
    long_parent = _row("s::a", content="x" * (transcript_parser.MAX_TURN_CHARS + 500))
    out = rag_engine._collapse_passages([long_parent])
    assert len(out[0]["content"]) < transcript_parser.MAX_TURN_CHARS + 100
    assert "get_turns" in out[0]["content"]


def test_search_output_names_the_passage():
    text = tools.format_results([{
        "doc_id": "s::a::p1", "chunk_type": PASSAGE_CHUNK_TYPE, "session_id": "s",
        "turn_index": 7, "content": "passage text", "passage_index": 1, "distance": 0.2,
    }])
    assert "part 2" in text
    assert "get_turns" in text


# --- listing and get_turns ----------------------------------------------------


def test_listing_skips_passages(monkeypatch):
    seen = {}

    def _batches(fields, filter_expr=None, db_path=None, **kw):
        seen["filter"] = filter_expr
        return iter(())

    monkeypatch.setattr(rag_engine, "_query_batches", _batches)
    rag_engine._recent_listing(5, provider="codex", db_path=DB)
    assert 'chunk_type != "turn_passage"' in seen["filter"]
    assert 'provider == "codex"' in seen["filter"]


def test_get_turns_returns_parents_only(monkeypatch):
    stored = [
        {"document": "turn one", "doc_id": "s::1", "session_id": "s", "turn_index": 1,
         "chunk_type": "turn"},
        {"document": "passage", "doc_id": "s::1::p0", "session_id": "s", "turn_index": 1,
         "chunk_type": PASSAGE_CHUNK_TYPE},
        {"document": "turn two", "doc_id": "s::2", "session_id": "s", "turn_index": 2,
         "chunk_type": "turn"},
    ]

    class _Client:
        def query(self, collection_name, filter, output_fields=None, limit=None):
            assert 'chunk_type != "turn_passage"' in filter
            return [dict(r) for r in stored if r["chunk_type"] != PASSAGE_CHUNK_TYPE]

    @contextlib.contextmanager
    def _client(db_path=None):
        yield _Client()

    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    turns = rag_engine.get_turns("s", 1, context=2, db_path=DB)
    assert [t["content"] for t in turns] == ["turn one", "turn two"]


def test_issue_timeline_skips_passages(monkeypatch):
    rows = [
        {"document": "SESF-1 turn", "doc_id": "s::1", "session_id": "s", "chunk_type": "turn",
         "timestamp": "2026-10-31T10:00:00Z", "issue_ids": ",SESF-1,", "provider": "codex"},
        {"document": "SESF-1 passage", "doc_id": "s::1::p0", "session_id": "s",
         "chunk_type": PASSAGE_CHUNK_TYPE, "timestamp": "2026-10-31T10:00:00Z",
         "issue_ids": ",SESF-1,", "provider": "codex"},
    ]
    monkeypatch.setattr(rag_engine, "_query_batches",
                        lambda fields, filter_expr=None, db_path=None, **kw: iter([rows]))
    monkeypatch.setattr(rag_engine, "_fts", types.SimpleNamespace(
        search=lambda *a, **k: [dict(r, content=r["document"]) for r in rows]))
    entries = rag_engine.get_issue_timeline("SESF-1", db_path=DB)
    assert [e["doc_id"] for e in entries] == ["s::1"]


def test_session_turns_and_day_rows_skip_passages(monkeypatch):
    seen = []

    def _batches(fields, filter_expr=None, db_path=None, **kw):
        seen.append(filter_expr)
        return iter(())

    monkeypatch.setattr(rag_engine, "_query_batches", _batches)
    rag_engine.session_turns("s", "codex", limit=5, db_path=DB)
    rag_engine._rows_on_day(__import__("datetime").date(2026, 10, 31), ["/p"], DB)
    assert all('chunk_type != "turn_passage"' in f for f in seen)


def test_stats_count_turns_without_passages(monkeypatch):
    rows = [{"session_id": "s", "chunk_type": "turn"},
            {"session_id": "s", "chunk_type": PASSAGE_CHUNK_TYPE},
            {"session_id": "s", "chunk_type": PASSAGE_CHUNK_TYPE}]
    lag_args = {}

    class _Client:
        def has_collection(self, name):
            return True

    @contextlib.contextmanager
    def _client(db_path=None):
        yield _Client()

    def _lag(db_path=None, project_root=None, milvus_turn_count=None):
        lag_args["count"] = milvus_turn_count
        return {"fts_row_count": 3, "fts_lag": 0, "fts_backfill_required": False}

    monkeypatch.setattr(rag_engine, "milvus_client", _client)
    monkeypatch.setattr(rag_engine, "_query_all", lambda *a, **k: rows)
    monkeypatch.setattr(rag_engine, "fts_lag_status", _lag)
    stats = rag_engine.get_stats(db_path=DB)
    assert stats["total_turns"] == 1
    assert stats["by_type"] == {"turn": 1, PASSAGE_CHUNK_TYPE: 2}
    assert lag_args["count"] == 3
