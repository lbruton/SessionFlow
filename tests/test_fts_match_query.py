"""SESF-93: keyword search builds its FTS5 MATCH string from terms.

Natural-language and mixed-identifier queries must reach the keyword channel,
and transcript and memory search must build the same MATCH string.
"""

import pytest

import fts_hybrid
from fts_hybrid import FTSIndex, build_match_query

PADDING_DOC = (
    "SESF-57 fixed the padding leak: batched embeddings drifted from single "
    "embeddings because the padding mask was additive."
)
OTHER_DOCS = [
    "The watcher parses transcripts mid-turn and keeps the last turn open.",
    "Milvus Standalone runs the HNSW index with COSINE distance.",
    "SESF-58 made index_state.json writes atomic; 57 files were affected.",
]


def _index(tmp_path, docs):
    fts = FTSIndex("turns_fts", ["session_id"])
    db_path = str(tmp_path / "milvus.db")
    conn = fts.connection(db_path)
    fts.insert(conn, [
        {"doc_id": f"d{i}", "content": text, "session_id": "s1"}
        for i, text in enumerate(docs)
    ])
    conn.commit()
    fts.close_ephemeral(conn)
    return fts, db_path


@pytest.mark.parametrize("query", [
    "padding",
    "Why do batched embeddings change with padding?",
    "SESF-57 padding",
])
def test_review_queries_find_the_padding_doc(tmp_path, query):
    fts, db_path = _index(tmp_path, [PADDING_DOC] + OTHER_DOCS)
    hits = fts.search(query, n=5, db_path=db_path)
    assert hits, f"no keyword hits for {query!r}"
    assert hits[0]["doc_id"] == "d0"


@pytest.mark.parametrize("query", ["", "   ", "?!", "the and of", "what is it?", '"'])
def test_queries_without_terms_return_nothing(tmp_path, query):
    fts, db_path = _index(tmp_path, [PADDING_DOC] + OTHER_DOCS + ["the and of what is it"])
    assert build_match_query(query) == ""
    assert fts.search(query, n=5, db_path=db_path) == []


def test_identifier_query_ranks_exact_identifier_first(tmp_path):
    docs = [
        "padding padding padding everywhere in this note",
        "SESF-58 notes mention 57 and sesf separately",
        "Shipped SESF-57 today",
    ]
    fts, db_path = _index(tmp_path, docs)
    hits = fts.search("SESF-57", n=5, db_path=db_path)
    assert [h["doc_id"] for h in hits] == ["d2"]


def test_identifier_tokens_stay_one_phrase():
    assert build_match_query("SESF-57 padding") == '"SESF-57" OR "padding"'
    assert build_match_query("rag_engine.py,") == '"rag_engine.py"'
    assert build_match_query("192.168.1.83") == '"192.168.1.83"'


def test_plain_words_split_on_punctuation_and_drop_stopwords():
    assert build_match_query("Why do batched embeddings change with padding?") == (
        '"batched" OR "embeddings" OR "change" OR "padding"'
    )
    assert build_match_query("mask/padding (leak)") == '"mask" OR "padding" OR "leak"'


def test_terms_are_deduplicated_case_insensitively():
    assert build_match_query("Padding padding PADDING mask") == '"Padding" OR "mask"'


def test_short_tokens_with_punctuation_split_into_words():
    assert build_match_query("a-b v1") == '"b" OR "v1"'


def test_double_quotes_inside_terms_cannot_break_syntax(tmp_path):
    fts, db_path = _index(tmp_path, [PADDING_DOC])
    for query in ['pad"ding', '"SESF-57"', 'x" OR "y', "NEAR(padding mask)", "padding*", "-padding"]:
        fts.search(query, n=5, db_path=db_path)


def test_pre_quoted_issue_id_still_matches(tmp_path):
    """The issue timeline passes a quoted canonical ID."""
    fts, db_path = _index(tmp_path, [PADDING_DOC] + OTHER_DOCS)
    hits = fts.search('"SESF-57"', n=5, db_path=db_path)
    assert [h["doc_id"] for h in hits] == ["d0"]


def test_memory_and_transcript_search_build_the_same_match(tmp_path, monkeypatch):
    from memory_index import MemoryFTS

    calls = []
    real = fts_hybrid.build_match_query

    def spy(query):
        out = real(query)
        calls.append((query, out))
        return out

    monkeypatch.setattr(fts_hybrid, "build_match_query", spy)

    query = "Why does SESF-57 change padding?"
    fts, db_path = _index(tmp_path, [PADDING_DOC])
    fts.search(query, n=5, db_path=db_path)
    MemoryFTS(tmp_path / "memories_fts.db").search(query, project="*", limit=5)

    assert len(calls) == 2
    assert calls[0] == calls[1]
