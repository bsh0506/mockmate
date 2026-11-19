"""FastAPI 后端：RAG 能力包成 REST API，业务数据落 MySQL（单用户，无鉴权）。"""
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session as OrmSession

from db.db import get_db, engine, Base
from ai_interviewer.models import ChatSession, Message, Interview
from schemas import (ChatRequest, ChatResponse, InterviewStartRequest,
                     InterviewStartResponse, InterviewChatRequest)
from rag.hybrid import create_assistant
from rag.interview import Interviewer
from fastapi.responses import StreamingResponse
assistant = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant
    Base.metadata.create_all(bind=engine)
    assistant = await create_assistant()
    yield
    assistant.close()
    await assistant.rag.finalize_storages()


app = FastAPI(title="MockMate API", lifespan=lifespan)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: OrmSession = Depends(get_db)):
    session = db.query(ChatSession).get(req.session_id) if req.session_id else None
    if not session:
        session = ChatSession(mode="qa", title=req.question[:30])
        db.add(session); db.commit(); db.refresh(session)

    t0 = time.time()
    answer, sources, path = await assistant.answer_with_meta(req.question)
    latency = int((time.time() - t0) * 1000)

    db.add(Message(session_id=session.id, role="user", question=req.question))
    db.add(Message(session_id=session.id, role="assistant", answer=answer,
                   sources=sources, latency_ms=latency))
    db.commit()
    return ChatResponse(session_id=session.id, answer=answer,
                        sources=sources, path=path, latency_ms=latency)

@app.post("/api/interview/chat")
def interview_chat(req: InterviewChatRequest, db: OrmSession = Depends(get_db)):
    session = db.query(ChatSession).get(req.session_id)
    if not session or session.mode != "interview":
        raise HTTPException(404, "会话不存在")
    record = (db.query(Interview).filter(Interview.session_id == session.id)
              .order_by(Interview.id.desc()).first())
    if not record:
        raise HTTPException(404, "尚未开始面试")

    history = [
        {"role": "user", "content": m.question}
        if m.role == "user" else {"role": "assistant", "content": m.answer}
        for m in (db.query(Message).filter(Message.session_id == session.id)
                  .order_by(Message.id).all())
    ]
    history.insert(0, {"role": "assistant", "content": record.question})  # 补第一题
    history.append({"role": "user", "content": req.answer})

    interviewer = Interviewer(assistant)
    reply = interviewer.chat({"text": record.knowledge}, history)

    db.add(Message(session_id=session.id, role="user", question=req.answer))
    db.add(Message(session_id=session.id, role="assistant", answer=reply))
    db.commit()
    return {"reply": reply}

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, db: OrmSession = Depends(get_db)):
    session = db.query(ChatSession).get(req.session_id) if req.session_id else None
    if not session:
        session = ChatSession(mode="qa", title=req.question[:30])
        db.add(session); db.commit(); db.refresh(session)

    t0 = time.time()

    async def event_stream():
        answer_parts = []
        meta = {"path": "traditional", "sources": []}
        async for item in assistant.answer_stream(req.question):
            kind, payload = item[0], item[1]
            if kind == "meta":
                meta = payload
                continue
            if kind == "answer":
                answer_parts.append(payload)
            yield f"data: {json.dumps({'type': kind, 'delta': payload}, ensure_ascii=False)}\n\n"

        latency = int((time.time() - t0) * 1000)
        answer = "".join(answer_parts)

        db.add(Message(session_id=session.id, role="user", question=req.question))
        db.add(Message(session_id=session.id, role="assistant", answer=answer,
                       sources=meta["sources"], latency_ms=latency))
        db.commit()

        yield f"data: {json.dumps({'type': 'done', 'session_id': session.id,
                                   'path': meta['path'], 'sources': meta['sources'],
                                   'latency_ms': latency}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/interview/chat/stream")
def interview_chat_stream(req: InterviewChatRequest, db: OrmSession = Depends(get_db)):
    session = db.query(ChatSession).get(req.session_id)
    if not session or session.mode != "interview":
        raise HTTPException(404, "会话不存在")
    record = (db.query(Interview).filter(Interview.session_id == session.id)
              .order_by(Interview.id.desc()).first())
    if not record:
        raise HTTPException(404, "尚未开始面试")

    history = [
        {"role": "user", "content": m.question}
        if m.role == "user" else {"role": "assistant", "content": m.answer}
        for m in (db.query(Message).filter(Message.session_id == session.id)
                  .order_by(Message.id).all())
    ]
    history.insert(0, {"role": "assistant", "content": record.question})  # 补第一题
    history.append({"role": "user", "content": req.answer})

    interviewer = Interviewer(assistant)

    def event_stream():
        reply_parts = []
        for kind, text in interviewer.chat_stream({"text": record.knowledge}, history):
            if kind == "answer":
                reply_parts.append(text)
            yield f"data: {json.dumps({'type': kind, 'delta': text}, ensure_ascii=False)}\n\n"
        reply = "".join(reply_parts)
        db.add(Message(session_id=session.id, role="user", question=req.answer))
        db.add(Message(session_id=session.id, role="assistant", answer=reply))
        db.commit()
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/sessions")
def list_sessions(mode: str | None = None, db: OrmSession = Depends(get_db)):
    q = db.query(ChatSession)
    if mode:
        q = q.filter(ChatSession.mode == mode)
    sessions = q.order_by(ChatSession.id.desc()).all()
    return [
        {"id": s.id, "mode": s.mode,
         "title": s.title or "未命名会话",
         "created_at": s.created_at.strftime("%m-%d %H:%M")}
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: int, db: OrmSession = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    msgs = (db.query(Message).filter(Message.session_id == session_id)
            .order_by(Message.id).all())
    result = [
        {"role": m.role, "content": m.question if m.role == "user" else m.answer}
        for m in msgs
    ]
    # 面试会话：面试官第一题存在 interviews 表里，补到最前面
    if session and session.mode == "interview":
        record = (db.query(Interview).filter(Interview.session_id == session_id)
                  .order_by(Interview.id.asc()).first())
        if record and record.question:
            result.insert(0, {"role": "assistant", "content": record.question})
    return result

@app.post("/api/interview/start", response_model=InterviewStartResponse)
def interview_start(req: InterviewStartRequest, db: OrmSession = Depends(get_db)):
    interviewer = Interviewer(assistant)
    question, knowledge = interviewer.ask(req.direction)

    # 标题 = 面试官第一题的前 30 字
    session = ChatSession(mode="interview", title=question[:30])
    db.add(session); db.commit(); db.refresh(session)

    db.add(Interview(session_id=session.id, question=question,
                     knowledge=knowledge["text"]))
    db.commit()
    return InterviewStartResponse(session_id=session.id, question=question)

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int, db: OrmSession = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    # 先删子记录（messages / interviews），再删会话，否则外键会报错
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.query(Interview).filter(Interview.session_id == session_id).delete()
    db.delete(session)
    db.commit()
    return {"ok": True}