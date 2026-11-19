from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: int | None = None
    question: str


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    sources: list[dict]
    path: str
    latency_ms: int


class InterviewStartRequest(BaseModel):
    direction: str = "RAG"


class InterviewStartResponse(BaseModel):
    session_id: int
    question: str


class InterviewChatRequest(BaseModel):
    session_id: int
    answer: str