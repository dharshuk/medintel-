# MedIntel

MedIntel is a full-stack multimodal medical assistant that blends a neon medical UI with multimodal tooling (voice, OCR, lab parsing, and routed LLM providers). The repo contains a React + Vite frontend and a FastAPI backend.

## Project Structure

```
frontend/   # React + Vite + Tailwind client
backend/    # FastAPI service with model routing + OCR/lab engines
```

## Frontend (React + Vite)

### Features
- ChatGPT/Perplexity-style layout with sidebar history, avatar, waveform, and glassmorphic bubbles.
- Mode selector (Medical, General, Mental Health, Student) that alters routing hints.
- Voice input (Web Speech API STT) and voice output (Speech Synthesis API TTS).
- File uploader (PDF/Image/Audio) pushing to `/api/upload`.
- Report side panel showing parsed labs + extracted text.

### Setup
```powershell
cd frontend
npm install
npm run dev
```
Vite proxies `/api/*` to `http://localhost:8000` by default (see `vite.config.js`).

### Build
```powershell
cd frontend
npm run build
npm run preview
```

## Backend (FastAPI)

### Features
- `/chat` routes questions through Groq, Gemini, or OpenAI based on mode/student hints.
- `/upload` ingests PDF/image/audio, runs EasyOCR (images) + lab regex parsing, and stores report snapshots.
- `/history` and `/report/{id}` expose in-memory chat/report data.
- Risk scoring + emotion detection placeholders keep responses safe.

### Setup
Create a virtual environment (recommended) and install requirements:
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the API:
```powershell
cd backend
uvicorn app.main:app --reload --port 8000
```

### Environment
Add your model credentials before enabling live completions:
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

Without keys, the router returns informative placeholder completions so the UI remains testable.

### OCR Notes
- EasyOCR requires `torch`. Installing the listed requirements pulls the CPU wheels automatically but may take a moment.
- Image bytes are decoded via Pillow + NumPy before reaching EasyOCR. Audio/PDF paths currently respond with descriptive placeholders awaiting future integrations.

## Connecting Frontend & Backend
1. Start FastAPI on `http://localhost:8000`.
2. Start Vite dev server (`npm run dev`).
3. Frontend requests to `/api/...` are proxied to the backend. Update `vite.config.js` if the backend address changes.

## Testing & Validation
- Frontend: `npm run build` (already executed successfully during scaffolding).
- Backend: `uvicorn app.main:app --reload` to verify endpoints once dependencies install.

## Next Steps
- Swap placeholder LLM calls with official Groq/Gemini/OpenAI SDK calls.
- Wire audio transcription (Whisper/Gemini) and real PDF parsing (e.g., pdfplumber + Gemini Vision).
- Persist chat history/report metadata to a database instead of in-memory structures.
- Add comprehensive unit tests for services (risk engine, lab parser, router, OCR).
