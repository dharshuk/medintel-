"""FastAPI application entry point for MedIntel."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .routes.chat import router as chat_router
from .routes.upload import router as upload_router

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="MedIntel API",
    description="Backend services for the MedIntel assistant",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "medintel"}
