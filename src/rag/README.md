# CrediX RAG Module (`src/rag`)

This package contains the core retrieval, search indexing, and validation interfaces for the CrediX offline-certified RAG system.

---

## 📁 Directory Structure

```text
src/
├── rag/
│   ├── __init__.py                    # Package exports and convenient imports
│   ├── bm25.py                        # Deterministic Arabic BM25 implementation
│   ├── retrieval_adapter.py           # Unified interface for Dense (FAISS) & Sparse (BM25) search
│   ├── eligibility.py                 # Conservative release blocking and candidate verification rules
│   ├── smoke_test.py                  # Smoke test runner for 5 core Arabic regulatory queries
│   ├── validate_release_candidate.py  # Production bundle validation (hashes, chunks, citations)
│   ├── requirements.txt               # Minimal Python dependencies for RAG execution
│   └── README.md                      # This documentation
│
└── rag_data/
    ├── current/                       # Active certified production release bundle
    │   ├── chunks.jsonl               # 384 human-reviewed regulatory chunks with page citations
    │   ├── parents.jsonl              # 85 parent provision/section records
    │   ├── bm25_corpus.jsonl          # Aligned tokenized corpus for BM25 lexical search
    │   ├── index.faiss                # FAISS IndexFlatIP (1024-dim BGE-M3 embeddings)
    │   ├── faiss_id_map.json          # Mapping of FAISS vector row IDs to chunk IDs
    │   ├── embedding_config.json      # BAAI/bge-m3 model configuration
    │   ├── search_index_config.json   # FAISS + BM25 hybrid search configuration
    │   ├── release_manifest.json      # Release metadata and SHA-256 integrity hashes
    │   └── documents_manifest.json    # Documents registry and source hashes
    │
    └── sources/                       # Official Central Bank of Egypt (CBE) PDF documents
        ├── DOC1_Credit_Granting_Regulations.pdf
        ├── DOC2_Creditworthiness_Assessment.pdf
        ├── DOC3_Credit_Registration_System.pdf
        ├── DOC4_MSME_Financing.pdf
        └── DOC5_Banking_Law_194_2020.pdf
```

---

## ⚡ Quick Start

### 1. Requirements
Install dependencies:
```bash
pip install -r src/rag/requirements.txt
```

### 2. Python Usage
```python
from src.rag.retrieval_adapter import dense_search, bm25_search, embed_query, get_parent_context

# 1. Embed query (using BGE-M3)
query = "ما هي شروط منح تسهيلات ائتمانية للعميل بالنقد الأجنبي؟"
query_vec = embed_query(query)

# 2. Dense semantic search
dense_results = dense_search(query_vec, top_k=3)
for hit in dense_results:
    print(f"[{hit['document_id']}] Score: {hit['score']:.4f}")
    print(f"Pages: {hit['metadata']['pdf_page_start']}-{hit['metadata']['pdf_page_end']}")
    print(hit["text_original"][:150] + "...")

# 3. Lexical BM25 search
bm25_results = bm25_search(query, top_k=3)

# 4. Fetch surrounding parent context
top_chunk = dense_results[0]
parent = get_parent_context(top_chunk["metadata"]["parent_id"])
print("Parent context:", parent["text_clean"][:200])
```

---

## 🧪 Validation & Tests

### Validate Release Bundle
```bash
python -m src.rag.validate_release_candidate --index-dir src/rag_data/current
```

### Run Smoke Test Suite
```bash
python -m src.rag.smoke_test --index-dir src/rag_data/current --require-citations
```
