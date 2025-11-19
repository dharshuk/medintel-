"""Upload endpoint handling OCR + lab parsing."""

from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.lab_parser import parse_labs_from_text
from ..services.ocr import extract_text
from ..services.state import REPORTS

router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload_report(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file received")

    payload = await file.read()
    extracted = await extract_text(file.filename, payload)
    labs = parse_labs_from_text(extracted)

    report_id = str(uuid4())
    report = {
        "id": report_id,
        "filename": file.filename,
        "extractedText": extracted,
        "labs": labs,
    }
    REPORTS[report_id] = report
    return report
