# Golden Retrieval v1

This dataset is chunk-level retrieval ground truth for the CrediX regulatory bundle. It measures whether Dense and BM25 retrieval return reviewed evidence chunks; it is not an LLM answer-quality dataset.

Labels were selected by inspecting `chunks.jsonl` and their parent records, rather than by treating a retriever's output as relevance. Answerable cases name the supported document, evidence chunk(s), and parent record(s). No-answer cases intentionally have no evidence labels and are reserved for a later evidence/no-answer evaluation stage.

Supported baseline metrics are Recall@1/3/5/10, MRR@10, nDCG@10, and HitRate@1/5/10, all at chunk level. Dense and BM25 are evaluated independently.

Run the baseline with:

```powershell
.venv-rag\Scripts\python.exe -m src.ai_assistant.evaluation.evaluate_retrieval
```

Additions must be reviewed against the source chunk and parent text, assigned a unique `query_id`, validated against the current bundle, and followed by an intentional dataset-version update.
