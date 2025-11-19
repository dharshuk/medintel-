"""Routing logic for selecting preferred foundation model provider."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

PROVIDER_MODELS = {
    "groq": "llama3-70b-8192",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_human_and_json(raw_text: str):
    """
    Returns (human_line, json_obj, raw_json_text)
    human_line = text BEFORE the first { ... }
    json_obj = parsed dict or None
    """
    if not raw_text:
        return ("", None, "")
    start = raw_text.find("{")
    if start == -1:
        # no JSON — entire text is human_line
        return (raw_text.strip(), None, "")
    human_line = raw_text[:start].strip()
    raw_json = raw_text[start: raw_text.rfind("}")+1]
    try:
        parsed = json.loads(raw_json)
    except Exception:
        parsed = None
    return (human_line, parsed, raw_json)


def dedupe_append(chat_list, role, text):
    """
    Add message to chat_list, avoiding duplicate consecutive messages
    chat_list: list of {"role":..., "text":...}
    """
    if not chat_list:
        chat_list.append({"role": role, "text": text})
        return
    last = chat_list[-1]
    if last["role"] == role and last["text"].strip() == text.strip():
        # skip duplicate
        return
    chat_list.append({"role": role, "text": text})


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class ModelPayload:
    question: str
    context: str
    provider_hint: Optional[str]
    student_mode: bool
    mode: str
    system_prompt: str


@dataclass
class ModelResult:
    provider: str
    answer: str
    summary: str
    citations: List[str]
    next_steps: List[str]


async def route_prompt(payload: ModelPayload) -> ModelResult:
    provider = _determine_provider(payload)
    answer = await _call_provider(provider, payload)
    
    # Extract first sentence for summary
    sentences = answer.replace('\n', ' ').split('. ')
    summary_text = sentences[0] + '.' if sentences else "AI analysis complete."
    if len(summary_text) > 150:
        summary_text = summary_text[:147] + "..."
    
    return ModelResult(
        provider=provider,
        answer=answer,
        summary=summary_text,
        citations=["Gemini AI Knowledge Base", "Medical Literature"],
        next_steps=[
            "Consult with a healthcare professional for diagnosis",
            "Track your symptoms in the MedIntel journal",
            "Share this summary with your doctor",
        ],
    )


def _determine_provider(payload: ModelPayload) -> str:
    if payload.provider_hint in PROVIDER_MODELS:
        return payload.provider_hint
    if payload.student_mode:
        return "openai"
    if payload.mode == "mental":
        return "groq"
    if any(ext in payload.context.lower() for ext in ["pdf", "image", "scan"]):
        return "gemini"
    return "groq"


async def _call_provider(provider: str, payload: ModelPayload) -> str:
    model_name = PROVIDER_MODELS.get(provider, PROVIDER_MODELS["groq"])
    
    # Build the full prompt
    prompt = (
        f"{payload.system_prompt}\n\n"
        f"Context: {payload.context}\n\n"
        f"Question: {payload.question}\n\n"
        f"Mode: {payload.mode}\n"
        f"Student mode: {payload.student_mode}"
    )

    # Try Gemini first (most reliable)
    if provider == "gemini" and genai:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini error: {type(e).__name__}: {e}")
                # Fall through to demo response
    
    # Try Groq
    if provider == "groq" and Groq:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                client = Groq(api_key=api_key)
                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": payload.system_prompt},
                        {"role": "user", "content": f"Context: {payload.context}\n\nQuestion: {payload.question}"}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Groq error: {e}")
    
    # Try OpenAI
    if provider == "openai" and OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": payload.system_prompt},
                        {"role": "user", "content": f"Context: {payload.context}\n\nQuestion: {payload.question}"}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"OpenAI error: {e}")
    
    # Fallback demo response
    return (
        f"**Demo Mode Active**\n\n"
        f"MedIntel would analyze your question using {provider.upper()} ({model_name}).\n\n"
        f"To enable live AI responses, set your API key:\n"
        f"- For Gemini: Set GEMINI_API_KEY environment variable\n"
        f"- For Groq: Set GROQ_API_KEY environment variable\n"
        f"- For OpenAI: Set OPENAI_API_KEY environment variable\n\n"
        f"Your question: {payload.question}"
    )
