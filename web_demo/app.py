import os
import uuid
import json
from typing import Optional
from pathlib import Path

import httpx
import aiofiles

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, StreamingResponse, Response

from utils.classifier import classify_image, get_classes
from utils.vlm_client import stream_analysis, vlm_client

# Configuration
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

VLM_HOST = os.getenv("VLM_HOST", "http://localhost:3026")

app = FastAPI(title="Skin AI Hub - Web Demo")

# CORS (allow localhost by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "classes": get_classes()}
    )


@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    """Classify the uploaded image and return predicted class + confidence"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Basic validation
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save to tmp
    filename = f"{uuid.uuid4().hex}_{file.filename or 'image.jpg'}"
    tmp_path = TMP_DIR / filename

    try:
        async with aiofiles.open(tmp_path, 'wb') as out:
            content = await file.read()
            await out.write(content)

        # Run classification
        predicted_class, confidence, probs = classify_image(str(tmp_path))

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
    finally:
        # Cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


@app.post("/generate_report")
async def generate_report_endpoint(request: Request):
    """
    Proxy streaming diagnosis from VLM server.

    Expected JSON body:
    {
      image_data: <base64 string>,
      patient_name: str,
      patient_age: int,
      patient_gender: str,
      classification: str,
      history: str,
      language: str (en|ko|vi),
      comprehensive: bool (optional)
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    image_b64: Optional[str] = body.get("image_data")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_data is required (base64)")

    patient_info = {
        "name": body.get("patient_name", "Unknown"),
        "age": body.get("patient_age", 30),
        "gender": body.get("patient_gender", "unknown"),
        "classification": body.get("classification", "Skin Condition"),
        "history": body.get("history", "No history provided"),
    }
    language = body.get("language", "en")
    comprehensive = bool(body.get("comprehensive", True))

    async def event_stream():
        try:
            # Stream tokens from VLM and convert to SSE-compatible lines for the browser
            if comprehensive:
                async for token in vlm_client.stream_diagnosis(image_b64, patient_info, language):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            else:
                async for token in vlm_client.stream_concise_analysis(image_b64, patient_info, language):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Explicit completion if upstream didn't send one
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    # Important: SSE content-type for streaming consumption
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/download_pdf")
async def download_pdf(path: str):
    """Proxy a report file from the VLM server given an API path (e.g. /api/v1/reports/foo.pdf)"""
    if not path.startswith("/api/"):
        raise HTTPException(status_code=400, detail="Invalid path")

    import httpx
    url = f"{VLM_HOST}{path}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "application/pdf")
            return Response(content=resp.content, media_type=content_type)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch PDF: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Optional: local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

