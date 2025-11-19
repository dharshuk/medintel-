# 🏥 MedIntel AI Medical Assistant

A warm, friendly, emotionally intelligent medical personal assistant powered by Google Gemini AI. Built with React + FastAPI, featuring a stunning glass morphism UI inspired by Perplexity, ChatGPT, and Apple Vision Pro.

## ✨ Features

- 🤖 **AI-Powered Chat**: Gemini 2.0-flash integration with warm, Gen-Z friendly tone
- 🎨 **Glass Morphism UI**: Modern, sleek interface with neon teal accents (#28F7CE)
- 🗣️ **Text-to-Speech**: Voice responses with play/stop controls
- 🎤 **Speech-to-Text**: Voice input support via Web Speech API
- 📊 **Medical Report Analysis**: Upload and analyze lab reports, medical documents
- 💾 **Session Management**: Persistent user profiles with conversation tracking
- 🔒 **Privacy-First**: In-memory session store, localStorage for preferences
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile

## 🚀 Tech Stack

### Frontend
- **React 18.3.1** - UI library
- **Vite 5.4.21** - Fast build tool
- **TailwindCSS 3.4.14** - Utility-first CSS with custom glass effects
- **Framer Motion 11.1.7** - Smooth animations
- **Axios 1.7.7** - API requests
- **Web Speech API** - TTS/STT capabilities

### Backend
- **FastAPI 0.115.2** - Modern Python web framework
- **Python 3.10** - Core language
- **Google Gemini AI** - Primary AI model (gemini-2.0-flash)
- **Groq** - Alternative LLM provider (llama3-70b-8192)
- **OpenAI** - GPT-4o-mini support
- **Uvicorn** - ASGI server

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Gemini API Key (from Google AI Studio)

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file in `backend/` directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here (optional)
OPENAI_API_KEY=your_openai_api_key_here (optional)
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

## 🎯 Running the Application

### Start Backend Server

```bash
cd backend
set KMP_DUPLICATE_LIB_OK=TRUE  # Windows (use export on Mac/Linux)
python -m uvicorn medintel_backend:app --reload --host 127.0.0.1 --port 8000
```

Backend will be available at: **http://127.0.0.1:8000**

### Start Frontend Server

```bash
cd frontend
npm run dev
```

Frontend will be available at: **http://127.0.0.1:5173**

## 🎭 Personality & Tone

MedIntel speaks like a caring Gen-Z friend who is:
- **Warm & Supportive**: "Hey love, I'm here. Tell me what's going on — I've got you 💛"
- **Smart & Clear**: Explains medical concepts in simple, friendly terms
- **Emotionally Intelligent**: Adapts to user's tone and emotional state
- **Safety-Conscious**: Recommends urgent care for serious symptoms

## 🔧 Project Structure

```
medintel2/
├── backend/
│   ├── medintel_backend.py      # Main FastAPI application
│   ├── user_store.py            # In-memory session management
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment variables (create this)
│   └── app/
│       ├── routes/              # API endpoints
│       │   ├── chat.py
│       │   └── upload.py
│       └── services/            # Business logic
│           ├── model_router.py  # AI model integration
│           ├── emotion.py       # Emotion detection
│           ├── lab_parser.py    # Medical report parsing
│           └── risk_engine.py   # Health risk assessment
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   └── ChatPage.jsx     # Main chat interface
│   │   ├── components/
│   │   │   ├── ChatBubble.jsx   # Message display with TTS
│   │   │   ├── ChatCenter.jsx   # Chat container
│   │   │   ├── Sidebar.jsx      # Navigation sidebar
│   │   │   └── RightPanel.jsx   # Info panel
│   │   ├── hooks/
│   │   │   ├── useLocalStore.js # localStorage management
│   │   │   ├── useTTS.js        # Text-to-speech
│   │   │   └── useSTT.js        # Speech-to-text
│   │   └── styles/
│   │       └── theme.css        # Custom CSS variables
│   ├── package.json
│   └── vite.config.js
│
└── README.md                    # This file
```

## 🌟 Key Features Explained

### Session Management
- **UUID-based sessions**: Each user gets a unique session ID stored in localStorage
- **Profile tracking**: Stores preferred tone, conversation count, health concerns
- **In-memory store**: Fast, privacy-focused data storage

### Smart Response Handling
- **Dual format support**: Handles both `{raw_text, human_line, json}` and structured medical data
- **Duplicate prevention**: Avoids repeating the same greeting in chat
- **Sequential TTS**: Speaks greeting first, then displays structured data

### Medical Safety
- ✅ Explains symptoms, conditions, lab results
- ✅ Provides mental health support and grounding techniques
- ✅ Offers lifestyle and health tips
- ❌ Does NOT diagnose or prescribe medication
- ⚠️ Recommends urgent care for serious symptoms

## 🎨 UI Highlights

- **Glass Morphism**: Frosted glass effects with backdrop blur
- **Neon Teal Theme**: Custom color palette (#28F7CE primary)
- **Smooth Animations**: Framer Motion for fluid transitions
- **Responsive Design**: Mobile-first approach with TailwindCSS
- **Dark Mode**: Sleek dark interface optimized for medical use

## 📡 API Endpoints

### POST `/api/v1/chat`
Send a message to the AI assistant
```json
{
  "question": "What does high cholesterol mean?",
  "context": [],
  "session_id": "uuid-here",
  "user_profile": {"preferred_tone": "warm_personal"}
}
```

### GET `/api/v1/profile/{session_id}`
Retrieve user profile

### POST `/api/v1/profile/{session_id}`
Update user profile

### POST `/api/v1/upload`
Upload medical documents for analysis

## 🔐 Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Required
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXX

# Optional (for multi-model support)
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXX
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX
```

## 🐛 Troubleshooting

### Backend won't start
- Ensure you're in the `backend/` directory
- Check that `.env` file exists with valid GEMINI_API_KEY
- Set `KMP_DUPLICATE_LIB_OK=TRUE` on Windows

### Frontend build errors
- Delete `node_modules/` and run `npm install` again
- Clear Vite cache: `rm -rf .vite/`
- Ensure Node.js version is 18+

### TTS not working
- Check browser permissions for audio
- Ensure browser supports Web Speech API (Chrome/Edge recommended)

## 📄 License

MIT License - feel free to use this project for learning or personal use.

## 🤝 Contributing

This is a personal project, but suggestions are welcome! Open an issue or submit a pull request.

## 💡 Acknowledgments

- **Google Gemini AI** - Powerful language model
- **TailwindCSS** - Beautiful styling system
- **Framer Motion** - Smooth animations
- Inspired by Perplexity, ChatGPT, and Apple Vision Pro design

---

Built with 💛 by **MedIntel Team**

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
