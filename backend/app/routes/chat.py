"""Chat orchestration endpoints with model routing + history."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.emotion import detect_emotion
from ..services.model_router import ModelPayload, route_prompt
from ..services.risk_engine import score_risk
from ..services.state import CHAT_HISTORY, REPORTS

SYSTEM_PROMPT = (
    "You are MedIntel — a safe, multimodal AI medical assistant.\n"
    "Rules:\n"
    "- Use ONLY provided context.\n"
    "- If unsure → say: ‘I don’t know; consult a clinician.’\n"
    "- Every response MUST include:\n"
    "  Summary, Answer, Risk Level, Confidence, Citations, Next Steps.\n"
    "- Avoid hallucination.\n"
    "- Student Mode → explain step-by-step.\n"
    "- Mental Health queries → respond calmly and safely."
)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., description="Primary user question")
    context: Optional[str] = Field(None, description="Report text or other context")
    model_provider: Optional[str] = Field(None, description="Preferred model vendor")
    student_mode: bool = Field(False, description="Whether to enable step-by-step answers")
    attachments: Optional[List[str]] = None
    mode: Optional[str] = Field("medical", description="UI-selected mode")


class ChatResponse(BaseModel):
    summary: str
    answer: str
    risk_level: str
    confidence: str
    emotion: str
    next_steps: List[str]
    citations: List[str]


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    model_payload = ModelPayload(
        question=payload.question,
        context=payload.context or "",
        provider_hint=payload.model_provider,
        student_mode=payload.student_mode,
        mode=payload.mode or "medical",
        system_prompt=SYSTEM_PROMPT,
    )

    llm_result = await route_prompt(model_payload)
    risk = score_risk(payload.question, llm_result.answer)
    emotion = detect_emotion(payload.question)

    response = ChatResponse(
        summary=llm_result.summary,
        answer=llm_result.answer,
        risk_level=risk.level,
        confidence=risk.confidence,
        emotion=emotion,
        next_steps=llm_result.next_steps,
        citations=llm_result.citations,
    )

    CHAT_HISTORY.appendleft({
        "question": payload.question,
        "answer": response.answer,
        "mode": payload.mode,
        "risk": response.risk_level,
    })

    return response


@router.get("/history")
async def history_endpoint():
    return list(CHAT_HISTORY)


@router.get("/report/{report_id}")
async def get_report(report_id: str):
    report = REPORTS.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
