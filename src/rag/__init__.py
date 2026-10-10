"""CrediX RAG Package."""
from .bm25 import BM25Index, normalize_for_bm25, tokenize, read_jsonl, write_jsonl
from .eligibility import release_blockers, release_blocking_chunks
from .release_integrity import ReleaseValidationError, validate_runtime_bundle

try:
    from .retrieval_adapter import dense_search, bm25_search, embed_query, get_chunk, get_parent_context
except ImportError:
    dense_search = None
    bm25_search = None
    embed_query = None
    get_chunk = None
    get_parent_context = None

__all__ = [
    "BM25Index",
    "normalize_for_bm25",
    "tokenize",
    "read_jsonl",
    "write_jsonl",
    "release_blockers",
    "release_blocking_chunks",
    "ReleaseValidationError",
    "validate_runtime_bundle",
    "dense_search",
    "bm25_search",
    "embed_query",
    "get_chunk",
    "get_parent_context",
]
