# ============================================================
#           MEDINTEL — ONE-FILE BACKEND (FULL VERSION)
#           Warm Personal Assistant Tone
# ============================================================

import os, json, io
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# AI Models
from groq import Groq
import google.generativeai as genai
from openai import OpenAI

# User profile store
from user_store import load_profile, save_profile, update_profile_field

# ------------------------------------------------------------
# INIT SERVICES
# ------------------------------------------------------------

app = FastAPI(title="MedIntel AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY   = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_KEY: 
    genai.configure(api_key=GEMINI_KEY)

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ------------------------------------------------------------
# WARM PERSONAL ASSISTANT SYSTEM PROMPT
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are MedIntel — a warm, friendly, caring personal medical assistant 💛

ANALYTICAL FRAMEWORK (Internal - Do NOT show these steps):
Before answering, think through:
1. FACTS: What relevant medical facts apply? What lab values/symptoms are present?
2. MECHANISMS: What biological processes are involved? Root causes?
3. ASSESSMENT: Analyze report values against normal ranges, identify patterns
4. RISK: What's the severity? Red flags? Timeline considerations?
5. ALTERNATIVES: What are treatment/management options? Lifestyle factors?
6. UNCERTAINTY: What needs clarification? What requires professional diagnosis?
7. MEMORY: Check user history - past symptoms, labs, patterns, preferences

OUTPUT: Only give the final answer - warm, clear, actionable guidance (NOT the analysis steps)

Tone:
- Speak like a gentle friend: calm, reassuring, empathetic
- Use simple words, clear explanations
- For casual greetings (hi, hello, hey): respond warmly in 1-2 short sentences
- Explain medical terms in everyday language

Response Guidelines:
- ALWAYS provide comprehensive, helpful answers to medical questions
- Use provided context when available, but also apply general medical knowledge
- Structure responses: Brief overview → detailed explanation → practical advice
- For symptoms: describe causes, severity indicators, when to seek care
- For conditions: explain what it is, common treatments, lifestyle tips
- For medications: explain purpose, how they work, common considerations
- Include relevant statistics or facts to support advice
- Reference USER_MEMORY (preferences, concerns) and MEDICAL_MEMORY (history, patterns) when available

Safety:
- For dangerous symptoms (chest pain, difficulty breathing, severe bleeding) → emphasize emergency care
- If diagnosis needed: recommend consulting healthcare provider
- Never replace professional medical diagnosis
- Be honest about uncertainty, but still provide useful general information

Format:
- Keep paragraphs short (2-3 sentences)
- Use bullet points for lists
- For greetings: just say hi warmly and ask how they're feeling
- For medical questions: provide thorough, evidence-based information
"""

# ------------------------------------------------------------
# FILE UPLOAD - Simplified
# ------------------------------------------------------------

def extract_text(file_bytes, filename):
    """Extract text from uploaded files"""
    ext = filename.lower().split(".")[-1]
    
    # Try to decode text files
    if ext in ["txt", "csv", "json", "md"]:
        try:
            return file_bytes.decode('utf-8', errors='ignore')
        except:
            return file_bytes.decode('latin-1', errors='ignore')
    
    # For PDFs and images, return placeholder but extract any available text
    if ext == "pdf":
        # Try simple text extraction
        try:
            text = file_bytes.decode('utf-8', errors='ignore')
            if len(text.strip()) > 20:
                return text
        except:
            pass
        return "[PDF file uploaded - please paste the text content or describe the report]"
    
    if ext in ["jpg","jpeg","png","bmp","gif"]:
        return "[Image file uploaded - please describe what the report shows or paste the text]"
    
    if ext in ["mp3","wav","m4a"]:
        return "[Audio file uploaded - please describe your symptoms or questions]"
    
    # Default: try to decode as text
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except:
        return "[File uploaded - please describe the content or paste any text from the report]"

# ------------------------------------------------------------
# SIMPLE LAB PARSER
# ------------------------------------------------------------

import re

def parse_labs(text):
    """Parse lab values from text with expanded pattern matching"""
    labs = []
    patterns = {
        "Hemoglobin": r"(?:hemoglobin|hb|hgb)[:\s\-]*([\d\.]+)\s*(?:g/dl|gm/dl)?",
        "WBC": r"(?:wbc|white blood cell)[:\s\-]*([\d\.]+)\s*(?:cells/cumm|x10\^3)?",
        "RBC": r"(?:rbc|red blood cell)[:\s\-]*([\d\.]+)\s*(?:million/cumm|x10\^6)?",
        "Platelets": r"(?:platelet|plt)[:\s\-]*([\d,\.]+)\s*(?:lakhs|x10\^3)?",
        "Blood Sugar": r"(?:blood sugar|glucose|fbs|fasting)[:\s\-]*([\d\.]+)\s*(?:mg/dl)?",
        "HbA1c": r"(?:hba1c|a1c|glycated)[:\s\-]*([\d\.]+)\s*(?:%|percent)?",
        "Cholesterol": r"(?:total cholesterol|cholesterol)[:\s\-]*([\d\.]+)\s*(?:mg/dl)?",
        "HDL": r"(?:hdl|good cholesterol)[:\s\-]*([\d\.]+)\s*(?:mg/dl)?",
        "LDL": r"(?:ldl|bad cholesterol)[:\s\-]*([\d\.]+)\s*(?:mg/dl)?",
        "Triglycerides": r"(?:triglycerides|tg)[:\s\-]*([\d\.]+)\s*(?:mg/dl)?",
        "Creatinine": r"(?:creatinine|creat)[:\s\-]*([\d\.]+)\s*(?:mg/dl)?",
        "TSH": r"(?:tsh|thyroid)[:\s\-]*([\d\.]+)\s*(?:miu/l|uiu/ml)?",
        "Vitamin D": r"(?:vitamin d|vit d|25-oh)[:\s\-]*([\d\.]+)\s*(?:ng/ml)?",
        "Vitamin B12": r"(?:vitamin b12|vit b12|b12)[:\s\-]*([\d\.]+)\s*(?:pg/ml)?",
    }
    lower = text.lower()
    for name, pat in patterns.items():
        m = re.search(pat, lower)
        if m:
            val = m.group(1).replace(",","")
            try:
                float_val = float(val)
                labs.append({"test_name": name, "value": val, "numeric_value": float_val})
            except:
                labs.append({"test_name": name, "value": val})
    return labs

# ------------------------------------------------------------
# RISK ENGINE
# ------------------------------------------------------------

def calculate_risk(labs):
    """Calculate risk level based on lab values with expanded parameters"""
    risk = "Green"
    issues = []
    
    for lab in labs:
        name = lab["test_name"]
        try:
            val = float(lab.get("numeric_value", lab["value"]))
        except:
            continue
            
        # Critical ranges
        if name == "Hemoglobin":
            if val < 7:
                risk = "Red"
                issues.append(f"Critically low hemoglobin: {val}")
            elif val < 10 and risk == "Green":
                risk = "Amber"
                issues.append(f"Low hemoglobin: {val}")
        
        elif name == "Blood Sugar":
            if val > 300:
                risk = "Red"
                issues.append(f"Critically high blood sugar: {val}")
            elif val > 180 and risk == "Green":
                risk = "Amber"
                issues.append(f"Elevated blood sugar: {val}")
        
        elif name == "HbA1c":
            if val > 9:
                risk = "Red"
                issues.append(f"Very high HbA1c: {val}%")
            elif val > 7 and risk == "Green":
                risk = "Amber"
                issues.append(f"Elevated HbA1c: {val}%")
        
        elif name == "Creatinine":
            if val > 3:
                risk = "Red"
                issues.append(f"High creatinine: {val}")
            elif val > 1.5 and risk == "Green":
                risk = "Amber"
                issues.append(f"Elevated creatinine: {val}")
        
        elif name == "Cholesterol":
            if val > 300 and risk == "Green":
                risk = "Amber"
                issues.append(f"High cholesterol: {val}")
        
        elif name == "WBC":
            if val > 15 or val < 3:
                if val > 20 or val < 2:
                    risk = "Red"
                    issues.append(f"Critical WBC: {val}")
                elif risk == "Green":
                    risk = "Amber"
                    issues.append(f"Abnormal WBC: {val}")
    
    return risk

# ------------------------------------------------------------
# EMOTION DETECTION
# ------------------------------------------------------------

def detect_emotion(text):
    t = text.lower()
    if any(x in t for x in ["sad","cry","hurt","depressed"]):
        return "sadness"
    if any(x in t for x in ["panic","anxious","scared","worry"]):
        return "anxiety"
    if any(x in t for x in ["angry","mad","furious"]):
        return "anger"
    return "neutral"

# ------------------------------------------------------------
# HELPER FUNCTIONS FOR JSON EXTRACTION
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# CHAT MODELS
# ------------------------------------------------------------

def call_gemini(prompt):
    if not GEMINI_KEY:
        return None
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text

def call_groq(prompt):
    if not groq_client:
        return None
    resp = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt}
        ],
        temperature=0.45
    )
    return resp.choices[0].message.content

def call_openai(prompt):
    if not openai_client:
        return None
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt}
        ],
        temperature=0.45
    )
    return resp.choices[0].message.content

# ------------------------------------------------------------
# API MODELS
# ------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = ""
    model_provider: Optional[str] = "auto"  # auto, gemini, groq, openai
    student_mode: Optional[bool] = False
    mode: Optional[str] = "medical"
    session_id: Optional[str] = None
    report_ids: Optional[List[int]] = None
    user_profile: Optional[dict] = None

class ChatResponse(BaseModel):
    summary: str
    answer: str
    risk_level: str
    confidence: str
    emotion: str
    next_steps: List[str]
    citations: List[str]
    model_used: Optional[str] = None  # Which AI model was used

# ------------------------------------------------------------
# CHAT ENDPOINT
# ------------------------------------------------------------

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    
    # Load user profile from in-memory store or use request profile as fallback
    if req.session_id:
        profile = load_profile(req.session_id)
        # Merge with any profile data sent in request
        if req.user_profile:
            profile.update(req.user_profile)
            save_profile(req.session_id, profile)
    else:
        # No session_id - use profile from request or default
        profile = req.user_profile or {
            "preferred_tone": "warm_personal",
            "name": "",
            "history_summary": ""
        }
    
    # Update conversation count if we have a session
    if req.session_id:
        profile['conversation_count'] = profile.get('conversation_count', 0) + 1
        save_profile(req.session_id, profile)
    
    # Build personal hint for the AI
    personal_hint = f"USER_PROFILE: {json.dumps(profile)}\n"
    
    # Extract memory information from profile
    user_memory = profile.get('preferences', {})
    medical_memory = {
        'history_summary': profile.get('history_summary', ''),
        'health_concerns': profile.get('health_concerns', []),
        'conversation_count': profile.get('conversation_count', 0)
    }
    
    # Build prompt with user profile and memory
    prompt = f"""
{personal_hint}

USER MEMORY:
{json.dumps(user_memory) if user_memory else "No user preferences recorded yet"}

MEDICAL MEMORY:
{json.dumps(medical_memory) if medical_memory.get('history_summary') or medical_memory.get('health_concerns') else "No medical history recorded yet"}

{SYSTEM_PROMPT}

CONTEXT:
{req.context if req.context else "No additional context provided."}

QUESTION:
{req.question}

MODE: {req.mode}
STUDENT_MODE: {req.student_mode}

Think through this step-by-step internally (facts, mechanisms, values, risk, alternatives, uncertainty), then provide a warm, friendly response with clear medical guidance.
"""

    # Intelligent model selection based on query characteristics
    def select_best_model(question: str, context: str, explicit_provider: str = None):
        """
        Smart model routing:
        - OpenAI (GPT-4o-mini): Complex reasoning, diagnosis, decision-making
        - Gemini (2.0-flash): General medical queries, explanations, conditions
        - Groq (llama3-70b): Quick responses, simple Q&A, greetings
        """
        if explicit_provider:
            return explicit_provider
        
        q_lower = question.lower()
        q_len = len(question)
        
        # Groq for speed: ONLY simple greetings (very short)
        greeting_only = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'bye', 'goodbye']
        if q_len < 25 and any(q_lower.strip() == word or q_lower.strip() == word + '!' for word in greeting_only):
            return "groq"
        
        # OpenAI for complex reasoning: Diagnosis, comparisons, decision-making
        reasoning_keywords = [
            'why do i', 'why am i', 'should i take', 'what if i', 
            'diagnose', 'which is better', 'compare', 'difference between',
            'decide', 'choose between', 'is it safe', 'can i take',
            'interaction', 'side effect'
        ]
        if any(keyword in q_lower for keyword in reasoning_keywords):
            return "openai"
        
        # Gemini for long docs: Extensive context or uploaded files
        if len(str(context)) > 5000:
            return "gemini"
        
        # Default to Gemini for ALL medical questions (most comprehensive)
        return "gemini"
    
    # Select model intelligently
    selected_model = select_best_model(req.question, str(req.context), req.model_provider if req.model_provider != "auto" else None)
    
    # Call the selected model with fallback to Gemini on error
    raw = None
    try:
        if selected_model == "gemini":
            raw = call_gemini(prompt)
        elif selected_model == "groq":
            raw = call_groq(prompt)
            # If Groq fails or returns None, fallback to Gemini
            if not raw:
                selected_model = "gemini"
                raw = call_gemini(prompt)
        elif selected_model == "openai":
            raw = call_openai(prompt)
            # If OpenAI fails or returns None, fallback to Gemini
            if not raw:
                selected_model = "gemini"
                raw = call_gemini(prompt)
        else:
            raw = call_gemini(prompt)  # Fallback
    except Exception as e:
        # On any error, fallback to Gemini
        print(f"Error with {selected_model}: {str(e)}")
        selected_model = "gemini"
        try:
            raw = call_gemini(prompt)
        except Exception as gemini_error:
            print(f"Gemini error: {str(gemini_error)}")
            raw = None

    if not raw:
        raw = "I'm having trouble connecting right now. Please try again in a moment 💙"

    # Try to extract human line and JSON
    human_line, parsed_json, raw_json = extract_human_and_json(raw)
    
    # Use human_line as summary if available, otherwise extract first sentence
    if human_line:
        summary = human_line[:150] if len(human_line) > 150 else human_line
    else:
        sentences = raw.replace('\n', ' ').split('. ')
        summary = sentences[0] + '.' if sentences else "Analysis complete"
        if len(summary) > 150:
            summary = summary[:147] + "..."

    # Detect emotion
    emotion = detect_emotion(req.question)
    
    # Calculate confidence (simple heuristic)
    confidence = "0.85" if req.context else "0.65"

    # Generate next steps
    next_steps = [
        "Share this information with your healthcare provider",
        "Track your symptoms in the MedIntel journal",
        "Schedule a follow-up if symptoms persist"
    ]

    # Citations
    citations = [f"{selected_model.upper()} AI Medical Knowledge", "Clinical Guidelines"]

    # Risk level
    risk_level = "Green"
    if any(word in req.question.lower() for word in ["severe", "emergency", "bleeding", "chest pain"]):
        risk_level = "Red"
        next_steps = ["⚠️ Seek emergency medical attention immediately", "Call emergency services if symptoms worsen"]
    elif any(word in req.question.lower() for word in ["pain", "worried", "concerned"]):
        risk_level = "Amber"

    return ChatResponse(
        summary=summary,
        answer=raw,
        risk_level=risk_level,
        confidence=confidence,
        emotion=emotion,
        next_steps=next_steps,
        citations=citations,
        model_used=selected_model  # Include which model was used
    )

# ------------------------------------------------------------
# USER PROFILE ENDPOINTS
# ------------------------------------------------------------

@app.get("/api/v1/profile/{session_id}")
def get_profile(session_id: str):
    """Get user profile by session ID"""
    profile = load_profile(session_id)
    return {"session_id": session_id, "profile": profile}


@app.post("/api/v1/profile/{session_id}")
def update_profile(session_id: str, profile_data: dict):
    """Update user profile"""
    success = save_profile(session_id, profile_data)
    if success:
        return {"status": "success", "profile": profile_data}
    else:
        raise HTTPException(status_code=500, detail="Failed to save profile")


@app.patch("/api/v1/profile/{session_id}/{field}")
def update_profile_field_endpoint(session_id: str, field: str, value: dict):
    """Update a specific field in user profile"""
    success = update_profile_field(session_id, field, value.get("value"))
    if success:
        return {"status": "success", "field": field, "value": value.get("value")}
    else:
        raise HTTPException(status_code=500, detail="Failed to update profile field")

# ------------------------------------------------------------
# UPLOAD ENDPOINT
# ------------------------------------------------------------

@app.post("/api/v1/upload")
async def upload(file: UploadFile = File(...)):
    """Upload and analyze medical reports with AI insights"""
    contents = await file.read()
    text = extract_text(contents, file.filename)
    labs = parse_labs(text)
    risk = calculate_risk(labs) if labs else "Green"
    
    # Generate AI analysis of the uploaded content
    analysis_prompt = f"""
Analyze this medical report and provide clear, helpful insights:

FILE: {file.filename}
CONTENT:
{text[:2000]}

LAB VALUES FOUND:
{json.dumps(labs, indent=2) if labs else 'No lab values detected'}

Please provide:
1. Summary of key findings
2. Explanation of any abnormal values in simple terms
3. Health recommendations based on the results
4. When to consult a doctor

Be warm, clear, and reassuring. Focus on actionable insights.
"""
    
    ai_analysis = ""
    try:
        # Use Gemini for comprehensive analysis
        ai_analysis = call_gemini(analysis_prompt)
    except Exception as e:
        print(f"AI analysis error: {str(e)}")
        ai_analysis = "I've received your file. " + (f"I found {len(labs)} lab values." if labs else "Please tell me what specific information you'd like me to explain from this report.")
    
    return {
        "filename": file.filename,
        "raw_text": text[:500] + "..." if len(text) > 500 else text,
        "extractedText": text,
        "parsed_labs": labs,
        "labs": labs,
        "risk_level": risk,
        "riskAssessment": risk,
        "ai_analysis": ai_analysis,
        "summary": f"Analyzed {file.filename} - Found {len(labs)} lab values" if labs else f"Received {file.filename}",
        "lab_count": len(labs)
    }

# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "medintel"}

@app.get("/")
def root():
    return {"status": "MedIntel Backend Running 💙", "version": "2.0"}
