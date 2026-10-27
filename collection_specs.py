"""Milvus collection definitions shared by every collection helper.

A ``CollectionSpec`` holds what create, ensure, drift detection and re-embed
need to know about one collection, so a later collection is added as a new spec
instead of new code paths. The module never imports ``rag_engine``, which
imports it; callers pass the embedding dimension of the active model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import DataType, FieldSchema

ISSUE_ID_FIELD_MAX = 4096
"""Byte cap of the comma-wrapped ``issue_ids`` field in every collection."""

SESSIONS = "sessions"
MEMORIES = "memories"


@dataclass(frozen=True)
class FieldSpec:
    """One typed Milvus field.

    Attributes:
        name: Field name.
        dtype: Milvus data type.
        max_length: VARCHAR byte limit; None for other types.
        dim: Vector dimension; None for scalar fields.
        is_primary: True for the INT64 primary key (never auto-generated).
    """

    name: str
    dtype: DataType
    max_length: Optional[int] = None
    dim: Optional[int] = None
    is_primary: bool = False

    def to_field_schema(self) -> FieldSchema:
        """Return the pymilvus ``FieldSchema`` for this field."""
        kwargs: Dict[str, Any] = {}
        if self.is_primary:
            kwargs.update(is_primary=True, auto_id=False)
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        if self.dim is not None:
            kwargs["dim"] = self.dim
        return FieldSchema(name=self.name, dtype=self.dtype, **kwargs)

    def signature(self) -> Tuple[str, int, Optional[int], Optional[int]]:
        """Return the comparable shape used by drift detection."""
        return (self.name, int(self.dtype), self.max_length, self.dim)


@dataclass(frozen=True)
class CollectionSpec:
    """Schema, vector index and read consistency of one collection.

    Attributes:
        name: Collection name.
        fields: Fields in schema order; exactly one is a vector field.
        vector_field: Name of the vector field the index covers.
        metric_type: Vector similarity metric.
        hnsw_params: HNSW build params used on Milvus Standalone.
        consistency_level: Read consistency set at creation, or None to keep
            the pymilvus default.
    """

    name: str
    fields: Tuple[FieldSpec, ...]
    vector_field: str = "vector"
    metric_type: str = "COSINE"
    hnsw_params: Dict[str, int] = field(
        default_factory=lambda: {"M": 16, "efConstruction": 256}
    )
    consistency_level: Optional[str] = None

    def field_schemas(self) -> List[FieldSchema]:
        """Return the pymilvus field list in schema order."""
        return [f.to_field_schema() for f in self.fields]

    def field_names(self) -> List[str]:
        """Return the field names in schema order."""
        return [f.name for f in self.fields]

    def vector_dim(self) -> int:
        """Return the dimension of the vector field."""
        for f in self.fields:
            if f.name == self.vector_field and f.dim is not None:
                return f.dim
        raise ValueError(f"{self.name}: no vector field {self.vector_field!r}")

    def index_spec(self, remote: bool) -> Dict[str, Any]:
        """Return the vector index definition for ``add_index``.

        Milvus Lite ignores every index type except FLAT, so only a remote
        (Standalone) server gets HNSW.
        """
        if remote:
            return {
                "field_name": self.vector_field,
                "index_type": "HNSW",
                "metric_type": self.metric_type,
                "params": dict(self.hnsw_params),
            }
        return {
            "field_name": self.vector_field,
            "index_type": "FLAT",
            "metric_type": self.metric_type,
        }


def sessions_spec(embed_dim: int) -> CollectionSpec:
    """Return the transcript ``sessions`` collection spec.

    Args:
        embed_dim: Vector dimension of the active embedding model.
    """
    return CollectionSpec(
        name=SESSIONS,
        fields=(
            FieldSpec("id", DataType.INT64, is_primary=True),
            FieldSpec("vector", DataType.FLOAT_VECTOR, dim=embed_dim),
            FieldSpec("document", DataType.VARCHAR, max_length=65535),
            FieldSpec("doc_id", DataType.VARCHAR, max_length=512),
            FieldSpec("session_id", DataType.VARCHAR, max_length=128),
            FieldSpec("logical_session_id", DataType.VARCHAR, max_length=256),
            FieldSpec("provider", DataType.VARCHAR, max_length=64),
            FieldSpec("source_kind", DataType.VARCHAR, max_length=96),
            FieldSpec("source_class", DataType.VARCHAR, max_length=32),
            FieldSpec("source_id", DataType.VARCHAR, max_length=512),
            FieldSpec("source_path", DataType.VARCHAR, max_length=1024),
            FieldSpec("transcript_file", DataType.VARCHAR, max_length=512),
            FieldSpec("turn_index", DataType.INT64),
            FieldSpec("timestamp", DataType.VARCHAR, max_length=64),
            FieldSpec("git_branch", DataType.VARCHAR, max_length=256),
            FieldSpec("chunk_type", DataType.VARCHAR, max_length=64),
            FieldSpec("project_root", DataType.VARCHAR, max_length=512),
            FieldSpec("issue_ids", DataType.VARCHAR, max_length=ISSUE_ID_FIELD_MAX),
        ),
    )


def memories_spec(embed_dim: int) -> CollectionSpec:
    """Return the ``memories`` index collection spec.

    The collection holds one vector and the filter scalars per memory. Text
    lives in Postgres and the local FTS file, so search validates every hit
    against Postgres; ``version`` and ``generation`` make that check possible.

    Args:
        embed_dim: Vector dimension of the active embedding model.
    """
    return CollectionSpec(
        name=MEMORIES,
        fields=(
            FieldSpec("id", DataType.INT64, is_primary=True),
            FieldSpec("memory_id", DataType.VARCHAR, max_length=36),
            FieldSpec("vector", DataType.FLOAT_VECTOR, dim=embed_dim),
            FieldSpec("user_id", DataType.VARCHAR, max_length=128),
            FieldSpec("agent_id", DataType.VARCHAR, max_length=64),
            FieldSpec("project", DataType.VARCHAR, max_length=128),
            FieldSpec("type", DataType.VARCHAR, max_length=32),
            FieldSpec("status", DataType.VARCHAR, max_length=16),
            FieldSpec("issue_ids", DataType.VARCHAR, max_length=ISSUE_ID_FIELD_MAX),
            FieldSpec("source", DataType.VARCHAR, max_length=128),
            FieldSpec("category", DataType.VARCHAR, max_length=128),
            FieldSpec("created_at", DataType.VARCHAR, max_length=64),
            FieldSpec("version", DataType.INT64),
            FieldSpec("generation", DataType.INT64),
        ),
        consistency_level="Strong",
    )
