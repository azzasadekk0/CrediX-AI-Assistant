"""Small SQLAlchemy persistence layer owned by the AI Assistant service."""

import secrets
from typing import Generator

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from .config import settings


database_url = settings.database_url
# Railway commonly exposes ``postgresql://``; SQLAlchemy should use the
# installed psycopg3 driver explicitly rather than looking for psycopg2.
if database_url.startswith("postgres://"):
    database_url = "postgresql+psycopg://" + database_url.removeprefix("postgres://")
elif database_url.startswith("postgresql://"):
    database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "ai_chat_sessions"

    id = Column(String(80), primary_key=True)
    user_id = Column(String(200), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    title_en = Column(String(200), nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "ai_chat_messages"

    id = Column(String(80), primary_key=True)
    session_id = Column(String(80), ForeignKey("ai_chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    sender = Column(String(20), nullable=False)
    text = Column(Text, nullable=False)
    text_en = Column(Text, nullable=True)
    timestamp = Column(String(50), nullable=False)
    citations = Column(JSON, nullable=False, default=list)
    answer_mode = Column(String(30), nullable=True)
    provenance = Column(String(30), nullable=True)
    disclaimer = Column(Text, nullable=True)
    segments = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False)
    session = relationship("ChatSession", back_populates="messages")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"
