from datetime import datetime
from sqlalchemy import (BigInteger, Column, String, Text, Integer,
                        DateTime, ForeignKey, Numeric)
from sqlalchemy.dialects.mysql import JSON
from db.db import Base


class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    mode = Column(String(16), nullable=False, default="qa")
    title = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.now)


class Message(Base):
    __tablename__ = "messages"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    question = Column(Text)
    answer = Column(Text)
    sources = Column(JSON)
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class Interview(Base):
    __tablename__ = "interviews"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id"), nullable=False, index=True)
    question = Column(Text)
    knowledge = Column(Text)
    answer = Column(Text)
    score = Column(Numeric(3, 1))
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.now)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    params = Column(JSON)
    hit_rate = Column(Numeric(5, 4))
    coverage = Column(Numeric(5, 4))
    created_at = Column(DateTime, default=datetime.now)