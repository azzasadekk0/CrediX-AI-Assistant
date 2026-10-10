# CrediX AI Assistant Integration Guide

This document is the integration contract for connecting the CrediX frontend/backend to the standalone AI Assistant service deployed on Railway.

## 1. Service information

Production service:

```text
https://credix-ai-assistant-production.up.railway.app
```

API base URL:

```text
https://credix-ai-assistant-production.up.railway.app/api/v1
```

Interactive API documentation:

```text
https://credix-ai-assistant-production.up.railway.app/docs
```

Health check:

```text
https://credix-ai-assistant-production.up.railway.app/health
```

The service is currently deployed as `credix-ai-assistant` in Railway.

## 2. Authentication

All AI Assistant endpoints require the logged-in user's Supabase access token. The health endpoint is public.

Send this header on every session and message request:

```http
Authorization: Bearer <SUPABASE_ACCESS_TOKEN>
Content-Type: application/json
```

The service reads the authenticated user's Supabase `sub` claim. Sessions and messages are isolated per user; a user cannot read or delete another user's session.

Example token helper in the frontend:

```ts
const { data } = await supabase.auth.getSession();
const accessToken = data.session?.access_token;

if (!accessToken) {
  throw new Error("The user is not authenticated");
}

const headers = {
  Authorization: `Bearer ${accessToken}`,
  "Content-Type": "application/json",
};
```

Never send the Supabase service-role key or a Gemini API key to the browser.

## 3. Answer modes

The UI labels map to API values as follows:

| UI label | API `mode` | Behavior |
|---|---|---|
| Auto (recommended) | `auto` | Routes the question between grounded, general, and hybrid handling. |
| Documents | `grounded` | Answers from the indexed CrediX documents. If evidence is insufficient, the service fails closed. |
| General AI | `general` | Uses the general LLM path and marks the answer as AI-generated. |

The service never accepts `documents` or `general_ai` as mode values. Convert the UI value before sending the request.

## 4. API endpoints

### 4.1 Health check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "credix-ai-assistant"
}
```

### 4.2 List the current user's chat sessions

```http
GET /api/v1/ai-assistant/sessions
```

Response:

```json
[
  {
    "id": "sess_123",
    "title": "تحليل طلب تمويل",
    "titleEn": "Financing Analysis",
    "active": true
  }
]
```

### 4.3 Create a chat session

```http
POST /api/v1/ai-assistant/sessions
```

Request body:

```json
{
  "title": "تحليل طلب تمويل",
  "titleEn": "Financing Analysis"
}
```

`titleEn` is optional. `title` is required.

Response status: `201 Created`

```json
{
  "id": "sess_123",
  "title": "تحليل طلب تمويل",
  "titleEn": "Financing Analysis",
  "active": true
}
```

### 4.4 Get messages for a session

```http
GET /api/v1/ai-assistant/sessions/{session_id}/messages
```

Response:

```json
[
  {
    "id": "msg_1",
    "sender": "user",
    "text": "ما هي شروط تقييم الجدارة الائتمانية؟",
    "textEn": "What are the creditworthiness assessment criteria?",
    "timestamp": "الآن",
    "citations": [],
    "answerMode": null,
    "provenance": null,
    "disclaimer": null,
    "segments": []
  }
]
```

### 4.5 Send a message

```http
POST /api/v1/ai-assistant/sessions/{session_id}/messages
```

Request body:

```json
{
  "text": "ما هي شروط تقييم الجدارة الائتمانية؟",
  "textEn": "What are the creditworthiness assessment criteria?",
  "mode": "auto"
}
```

Fields:

| Field | Required | Values |
|---|---:|---|
| `text` | Yes | User question, normally Arabic or English. |
| `textEn` | No | Optional English equivalent. If omitted, `text` is used. |
| `mode` | No | `auto`, `grounded`, or `general`; default is `auto`. |

The endpoint returns two messages: the saved user message and the generated assistant message.

```json
[
  {
    "id": "msg_user_123",
    "sender": "user",
    "text": "ما هي شروط تقييم الجدارة الائتمانية؟",
    "textEn": "What are the creditworthiness assessment criteria?",
    "timestamp": "الآن",
    "citations": [],
    "answerMode": null,
    "provenance": null,
    "disclaimer": null,
    "segments": []
  },
  {
    "id": "msg_bot_123",
    "sender": "assistant",
    "text": "الإجابة المبنية على المستندات...",
    "textEn": "The answer grounded in the documents...",
    "timestamp": "الآن",
    "citations": [
      {
        "documentName": "DOC2_Creditworthiness_Assessment.pdf",
        "documentNameEn": "Creditworthiness Assessment",
        "page": 4,
        "quote": "..."
      }
    ],
    "answerMode": "grounded",
    "provenance": "retrieved",
    "disclaimer": null,
    "segments": [
      {
        "text": "النص المدعوم من المستندات...",
        "sourceType": "retrieved",
        "citationHandles": ["C1"],
        "supportStatus": "supported"
      }
    ]
  }
]
```

### 4.6 Delete a chat session

```http
DELETE /api/v1/ai-assistant/sessions/{session_id}
```

Response status: `204 No Content`.

The frontend should remove the deleted session from its local list and select another session if one exists.

## 5. Understanding the assistant response

The frontend must use `answerMode`, `provenance`, `citations`, and `segments` to explain where the answer came from.

### `answerMode`

Possible values:

```text
grounded              Answer based on retrieved document evidence.
general               General LLM answer; not based on retrieved documents.
hybrid                Contains both retrieved evidence and general AI content.
insufficient_evidence The documents do not safely support an answer.
```

### `provenance`

Possible values:

```text
retrieved      Evidence came from the indexed documents.
ai_generated   Answer came from the general LLM path.
mixed          Answer contains both document evidence and general AI content.
unavailable    The service could not safely produce an answer.
```

### Rendering rules

1. `grounded` + `retrieved`: display document citations.
2. `general` + `ai_generated`: display a clear label such as `AI-generated / General knowledge`; do not display document citations.
3. `hybrid` + `mixed`: keep document-supported and general segments visually separated. Use citations only for segments with `sourceType: "retrieved"`.
4. `insufficient_evidence` or `unavailable`: display the safe fallback message and do not invent citations.
5. For each `segments` item, display `supportStatus` when the UI exposes claim-level evidence:
   - `supported`: supported by the cited document evidence.
   - `inference`: an explicitly marked inference.
   - `unsupported`: do not present it as a document fact.

The assistant response must not be re-written by the frontend in a way that removes its provenance or citations.

## 6. Frontend request flow

Recommended flow for the AI Assistant screen:

```text
1. Get the Supabase session access token.
2. GET /ai-assistant/sessions.
3. If there is no selected session, POST /ai-assistant/sessions.
4. Render GET /sessions/{id}/messages.
5. Map the selected UI mode to auto/grounded/general.
6. POST /sessions/{id}/messages.
7. Append the returned user and assistant messages.
8. Render provenance and citations from the assistant message.
9. DELETE the session when the user confirms deletion.
```

Use the same `session_id` for the whole conversation. Do not create a new session for every message.

## 7. TypeScript helper example

```ts
const AI_API_BASE =
  process.env.NEXT_PUBLIC_AI_ASSISTANT_API_URL ??
  "https://credix-ai-assistant-production.up.railway.app/api/v1";

async function aiRequest<T>(
  path: string,
  accessToken: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${AI_API_BASE}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`AI Assistant ${response.status}: ${body}`);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const listSessions = (token: string) =>
  aiRequest<Session[]>("/ai-assistant/sessions", token);

export const createSession = (token: string, title: string, titleEn?: string) =>
  aiRequest<Session>("/ai-assistant/sessions", token, {
    method: "POST",
    body: JSON.stringify({ title, titleEn }),
  });

export const listMessages = (token: string, sessionId: string) =>
  aiRequest<Message[]>(`/ai-assistant/sessions/${sessionId}/messages`, token);

export const sendMessage = (
  token: string,
  sessionId: string,
  text: string,
  mode: "auto" | "grounded" | "general",
  textEn?: string,
) =>
  aiRequest<Message[]>(`/ai-assistant/sessions/${sessionId}/messages`, token, {
    method: "POST",
    body: JSON.stringify({ text, textEn, mode }),
  });

export const deleteSession = (token: string, sessionId: string) =>
  aiRequest<void>(`/ai-assistant/sessions/${sessionId}`, token, {
    method: "DELETE",
  });
```

## 8. curl smoke tests

Replace `<TOKEN>` with a real Supabase user access token and `<SESSION_ID>` with a session ID.

```bash
# Health
curl https://credix-ai-assistant-production.up.railway.app/health

# List sessions
curl -H "Authorization: Bearer <TOKEN>" \
  https://credix-ai-assistant-production.up.railway.app/api/v1/ai-assistant/sessions

# Create session
curl -X POST \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"title":"اختبار المساعد","titleEn":"Assistant Test"}' \
  https://credix-ai-assistant-production.up.railway.app/api/v1/ai-assistant/sessions

# Send an Auto question
curl -X POST \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"text":"ما هي شروط تقييم الجدارة الائتمانية؟","mode":"auto"}' \
  https://credix-ai-assistant-production.up.railway.app/api/v1/ai-assistant/sessions/<SESSION_ID>/messages

# Delete session
curl -X DELETE \
  -H "Authorization: Bearer <TOKEN>" \
  https://credix-ai-assistant-production.up.railway.app/api/v1/ai-assistant/sessions/<SESSION_ID>
```

## 9. Error handling

| Status | Meaning | Frontend action |
|---:|---|---|
| `401` | Missing or invalid Supabase bearer token. | Refresh the auth session or redirect to login. |
| `404` | Session does not exist or belongs to another user. | Remove stale session locally and reload the session list. |
| `422` | Invalid request body or unsupported mode. | Fix the payload; mode must be `auto`, `grounded`, or `general`. |
| `502` | AI Assistant runtime/provider failed. | Show a retry message; do not fabricate an answer. |
| `204` | Session deleted successfully. | Remove it from the sidebar. |

Example safe error handling:

```ts
if (response.status === 401) {
  // Refresh Supabase session or redirect to login.
} else if (response.status === 404) {
  // The selected chat was deleted or is stale.
} else if (response.status === 502) {
  // Show "Assistant temporarily unavailable" and allow retry.
}
```

## 10. Environment variables for the integrating application

Frontend:

```env
NEXT_PUBLIC_AI_ASSISTANT_API_URL=https://credix-ai-assistant-production.up.railway.app/api/v1
```

If the existing frontend uses `NEXT_PUBLIC_API_URL`, point it to the AI service only for the AI Assistant API client, or use a separate variable so other CrediX APIs are not accidentally redirected.

Local frontend testing requires the Railway service CORS configuration to include `http://localhost:3000`. Production currently allows:

```text
https://credix-green.vercel.app
```

## 11. Important deployment note

The Railway service is currently using SQLite as a temporary database. It is suitable for connectivity and integration testing, but chat history can be lost after a container replacement or redeploy.

Before production use, set a persistent Postgres/Supabase `DATABASE_URL` in Railway and redeploy. Do not put database credentials, Supabase service-role keys, or Gemini keys in frontend code or Git.

## 12. Integration acceptance checklist

- [ ] Frontend obtains a Supabase access token before calling the service.
- [ ] Every request sends `Authorization: Bearer <token>`.
- [ ] Frontend maps Auto/Documents/General AI to `auto`/`grounded`/`general`.
- [ ] One session is reused for all messages in a conversation.
- [ ] `answerMode` and `provenance` are displayed to the user.
- [ ] Grounded citations are rendered from `citations`, not generated by the frontend.
- [ ] Hybrid segments remain separated by `sourceType` and `supportStatus`.
- [ ] `401`, `404`, `422`, and `502` are handled explicitly.
- [ ] Session deletion uses the `DELETE` endpoint and updates the sidebar.
- [ ] A persistent Postgres/Supabase database is configured before production.
- [ ] The production frontend origin is present in `CORS_ORIGINS`.
