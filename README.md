# CrediX AI Assistant — Railway Service

This folder is a self-contained deployment unit for the CrediX Assistant. It
contains the FastAPI chat API, Supabase JWT verification, chat persistence,
the frozen retrieval/generation runtime, and the current RAG release bundle.

## Railway setup

1. Create a Railway service from this folder (or copy this folder into its own
   repository).
2. Use the included `Dockerfile`.
3. Add every variable from `.env.example` in Railway. Keep `GEMINI_API_KEY`
   server-side only.
4. Use PostgreSQL for `DATABASE_URL` in production. SQLite is only for local
   smoke tests.
5. Set `CREDIX_RERANKER_DEVICE=cpu` unless the Railway service has a verified
   CUDA runtime and the matching model packages.

## API contract

All routes below require `Authorization: Bearer <Supabase access token>`:

- `GET /api/v1/ai-assistant/sessions`
- `POST /api/v1/ai-assistant/sessions`
- `GET /api/v1/ai-assistant/sessions/{session_id}/messages`
- `POST /api/v1/ai-assistant/sessions/{session_id}/messages`
- `DELETE /api/v1/ai-assistant/sessions/{session_id}`

The message body is:

```json
{"text":"...","textEn":"...","mode":"auto"}
```

`mode` is one of `auto`, `grounded`, or `general`. Responses expose
`answerMode`, `provenance`, `citations`, `segments`, and `disclaimer`, so the
frontend can clearly distinguish retrieved evidence from AI-generated text.

## Local smoke test

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Health check: `GET /health`.
