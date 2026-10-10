"""Standalone CrediX AI Assistant API for Railway deployment."""

from datetime import datetime, timezone
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .assistant import answer, citations_for
from .auth import require_user
from .config import settings
from .database import ChatMessage, ChatSession, get_db, initialize_database, new_id
from .schemas import ChatMessageCreate, MessageResponse, SessionCreate, SessionResponse


app = FastAPI(title=settings.project_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.normalized_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    initialize_database()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _session_or_404(db: Session, session_id: str, user_id: str) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


def _message_response(message: ChatMessage) -> dict:
    return {
        "id": message.id, "sender": message.sender, "text": message.text,
        "textEn": message.text_en, "timestamp": message.timestamp,
        "citations": message.citations or [], "answerMode": message.answer_mode,
        "provenance": message.provenance, "disclaimer": message.disclaimer,
        "segments": message.segments or [],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "credix-ai-assistant"}


@app.get("/api/v1/ai-assistant/sessions", response_model=list[SessionResponse])
def list_sessions(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    return db.query(ChatSession).filter(ChatSession.user_id == user["sub"]).order_by(ChatSession.created_at.desc()).all()


@app.post("/api/v1/ai-assistant/sessions", response_model=SessionResponse, status_code=201)
def create_session(payload: SessionCreate, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    now = _now()
    session = ChatSession(id=new_id("sess"), user_id=user["sub"], title=payload.title,
                          title_en=payload.title_en or payload.title, active=True,
                          created_at=now, updated_at=now)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/api/v1/ai-assistant/sessions/{session_id}/messages", response_model=list[MessageResponse])
def list_messages(session_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    session = _session_or_404(db, session_id, user["sub"])
    return [_message_response(message) for message in session.messages]


@app.post("/api/v1/ai-assistant/sessions/{session_id}/messages", response_model=list[MessageResponse])
def post_message(session_id: str, payload: ChatMessageCreate, response: Response,
                 db: Session = Depends(get_db), user: dict = Depends(require_user)):
    session = _session_or_404(db, session_id, user["sub"])
    response.headers["X-CrediX-Requested-Mode"] = payload.mode
    user_message = ChatMessage(id=new_id("msg_user"), session_id=session.id, sender="user",
                               text=payload.text, text_en=payload.text_en or payload.text,
                               timestamp="الآن", citations=[], segments=[], created_at=_now())
    db.add(user_message)
    try:
        result = answer(payload.text.strip(), payload.mode)
        if payload.mode == "grounded":
            answer_mode = "insufficient_evidence" if result.no_answer else "grounded"
            provenance = "retrieved"
            disclaimer = None
        else:
            answer_mode = result.answer_mode
            provenance = result.provenance
            disclaimer = getattr(result, "disclaimer", None)
        citations = citations_for(result) if answer_mode in {"grounded", "hybrid"} else []
        segments = [
            {"text": item.text, "sourceType": item.source_type,
             "citationHandles": list(item.citation_handles), "supportStatus": item.support_status}
            for item in getattr(result, "segments", [])
        ]
        assistant_message = ChatMessage(
            id=new_id("msg_bot"), session_id=session.id, sender="assistant", text=result.answer,
            text_en=result.answer, timestamp="الآن", citations=citations, answer_mode=answer_mode,
            provenance=provenance, disclaimer=disclaimer,
            segments=segments, created_at=_now(),
        )
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=502, detail="AI assistant runtime is unavailable") from error
    db.add(assistant_message)
    session.updated_at = _now()
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)
    return [_message_response(user_message), _message_response(assistant_message)]


@app.delete("/api/v1/ai-assistant/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    session = _session_or_404(db, session_id, user["sub"])
    db.delete(session)
    db.commit()
