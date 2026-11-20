import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.ingestion import UPLOADS_DIR, IngestionService
from app.services.retrieval import RetrievalService

_API_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _API_DIR.parent.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_API_DIR / ".env")

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ingestion_service: Optional[IngestionService] = None
_retrieval_service: Optional[RetrievalService] = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService(get_ingestion_service())
    return _retrieval_service


def sanitize_filename(filename: str) -> str:
    name = os.path.basename(filename or "document.pdf")
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []
    document: Optional[str] = None


class Citation(BaseModel):
    text: str
    page: int
    source: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}


@app.get("/health")
async def health():
    ingestion = get_ingestion_service()
    docs = ingestion.list_documents()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

    return {
        "status": "ok",
        "documents_count": len(docs),
        "total_chunks": sum(d.get("chunks", 0) for d in docs),
        "storage_backend": ingestion.last_storage_backend,
        "embedding_mode": ingestion.last_embedding_mode,
        "config": {
            "supabase_url_configured": bool(supabase_url),
            "supabase_service_key_looks_publishable": supabase_key.startswith("sb_publishable_"),
            "google_api_key_configured": bool(google_key),
            "google_api_key_looks_valid": google_key.startswith("AIza"),
        },
    }


@app.get("/documents")
async def list_documents():
    ingestion = get_ingestion_service()
    return {"documents": ingestion.list_documents()}


@app.get("/documents/{filename}/file")
async def get_document_file(filename: str):
    clean_name = sanitize_filename(filename)
    file_path = UPLOADS_DIR / clean_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF file '{clean_name}' not found on disk.")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{clean_name}"'},
    )


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    clean_name = sanitize_filename(filename)
    ingestion = get_ingestion_service()
    result = ingestion.delete_document(clean_name)
    return {"message": f"Deleted {clean_name}", **result}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    clean_name = sanitize_filename(file.filename)
    saved_path = UPLOADS_DIR / clean_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

    with open(saved_path, "wb") as buffer:
        buffer.write(content)

    try:
        ingestion_service = get_ingestion_service()
        result = await ingestion_service.process_pdf(str(saved_path), original_filename=clean_name)
        return {
            "message": f"'{clean_name}' indexed successfully ({result['chunks']} chunks across {result['pages']} pages).",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        retrieval_service = get_retrieval_service()
        result = await retrieval_service.answer_question(
            question=request.message.strip(),
            history=request.history,
            document_filter=request.document,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
