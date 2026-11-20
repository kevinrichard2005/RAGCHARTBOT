import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase.client import Client, create_client

# Load .env from monorepo root or apps/api
_API_DIR = Path(__file__).resolve().parents[2]
_ROOT_DIR = _API_DIR.parent.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_API_DIR / ".env")

DATA_DIR = _API_DIR / "data"
UPLOADS_DIR = _API_DIR / "uploads"
LOCAL_STORE_PATH = DATA_DIR / "vector_store.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 768


def compute_local_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Deterministic 768-dim feature hashing embedding used as an automatic
    fallback if the Google Gemini API key is invalid or unreachable.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vec

    # Unigrams + bigrams for better semantic overlap
    features = list(tokens) + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for feat in features:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class IngestionService:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

        self.supabase: Optional[Client] = None
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
            except Exception as e:
                print(f"[IngestionService] Supabase client init warning: {e}")

        self.embedding_models = ["models/text-embedding-004", "models/embedding-001"]
        self.last_embedding_mode = "gemini" if self._looks_like_gemini_key(self.google_api_key) else "local-fallback"
        self.last_storage_backend = "local"

    @staticmethod
    def _looks_like_gemini_key(key: str) -> bool:
        return bool(key) and key.startswith("AIza")

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], str]:
        """
        Embeds a list of texts using Gemini Embeddings, falling back cleanly
        to local 768-d embeddings if the API key is invalid or fails.
        """
        if self.google_api_key:
            for model_name in self.embedding_models:
                try:
                    embedder = GoogleGenerativeAIEmbeddings(
                        model=model_name,
                        google_api_key=self.google_api_key,
                    )
                    vectors = embedder.embed_documents(texts)
                    if vectors and len(vectors) == len(texts):
                        self.last_embedding_mode = f"gemini ({model_name})"
                        return vectors, self.last_embedding_mode
                except Exception as e:
                    print(f"[IngestionService] Gemini embedding ({model_name}) failed: {e}")

        self.last_embedding_mode = "local-fallback"
        return [compute_local_embedding(t) for t in texts], self.last_embedding_mode

    def embed_query(self, query: str) -> Tuple[List[float], str]:
        if self.google_api_key:
            for model_name in self.embedding_models:
                try:
                    embedder = GoogleGenerativeAIEmbeddings(
                        model=model_name,
                        google_api_key=self.google_api_key,
                    )
                    vec = embedder.embed_query(query)
                    if vec:
                        self.last_embedding_mode = f"gemini ({model_name})"
                        return vec, self.last_embedding_mode
                except Exception as e:
                    print(f"[IngestionService] Gemini query embedding ({model_name}) failed: {e}")

        self.last_embedding_mode = "local-fallback"
        return compute_local_embedding(query), self.last_embedding_mode

    def _load_local_store(self) -> List[Dict[str, Any]]:
        if not LOCAL_STORE_PATH.exists():
            return []
        try:
            with open(LOCAL_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_local_store(self, records: List[Dict[str, Any]]) -> None:
        with open(LOCAL_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

    async def process_pdf(self, file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts text from a PDF, chunks it with page metadata, deduplicates previous
        uploads of the same file, and stores vectors in Supabase (with local persistence).
        """
        clean_filename = original_filename or os.path.basename(file_path)

        # 1. Load PDF pages using PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        total_pages = len(documents)

        if not documents or all(not doc.page_content.strip() for doc in documents):
            raise ValueError(
                f"No extractable text found in '{clean_filename}'. "
                "Make sure the PDF contains selectable text (not scanned images without OCR)."
            )

        # 2. Split into chunks (~800 tokens / 1000 chars, 150 overlap)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)

        now_iso = datetime.now(timezone.utc).isoformat()
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            raw_page = chunk.metadata.get("page", 0)
            page_num = int(raw_page) + 1 if isinstance(raw_page, (int, float)) else 1
            meta = {
                "source": clean_filename,
                "page": page_num,
                "total_pages": total_pages,
                "chunk_index": idx,
                "uploaded_at": now_iso,
            }
            chunk.metadata = meta
            texts.append(chunk.page_content)
            metadatas.append(meta)

        # 3. Compute embeddings
        vectors, embedding_mode = self.embed_texts(texts)

        # 4. Always persist to local vector store (and deduplicate existing chunks for this source)
        existing_local = [
            r for r in self._load_local_store()
            if r.get("metadata", {}).get("source") != clean_filename
        ]
        new_records = [
            {
                "id": f"{clean_filename}:{idx}",
                "content": text,
                "metadata": meta,
                "embedding": vec,
            }
            for idx, (text, meta, vec) in enumerate(zip(texts, metadatas, vectors))
        ]
        self._save_local_store(existing_local + new_records)

        # 5. Also store in Supabase pgvector if configured and table exists
        supabase_stored = False
        supabase_error: Optional[str] = None
        if self.supabase is not None:
            try:
                # Delete prior chunks for the same filename to prevent duplicates
                self.supabase.table("documents").delete().contains(
                    "metadata", {"source": clean_filename}
                ).execute()

                rows = [
                    {
                        "content": text,
                        "metadata": meta,
                        "embedding": vec,
                    }
                    for text, meta, vec in zip(texts, metadatas, vectors)
                ]
                self.supabase.table("documents").insert(rows).execute()
                supabase_stored = True
                self.last_storage_backend = "supabase"
            except Exception as e:
                supabase_error = str(e)
                self.last_storage_backend = "local"
                print(f"[IngestionService] Supabase storage skipped/fallback to local: {e}")

        return {
            "filename": clean_filename,
            "chunks": len(chunks),
            "pages": total_pages,
            "storage_backend": "supabase" if supabase_stored else "local",
            "embedding_mode": embedding_mode,
            "supabase_warning": supabase_error,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lists all indexed documents with chunk counts, page counts, and timestamps.
        Merges records from local store and Supabase (if available).
        """
        docs_map: Dict[str, Dict[str, Any]] = {}

        for record in self._load_local_store():
            meta = record.get("metadata", {})
            source = meta.get("source", "unknown.pdf")
            entry = docs_map.setdefault(
                source,
                {
                    "filename": source,
                    "chunks": 0,
                    "pages": meta.get("total_pages", 1),
                    "uploaded_at": meta.get("uploaded_at"),
                    "has_file": (UPLOADS_DIR / source).exists(),
                },
            )
            entry["chunks"] += 1
            entry["pages"] = max(entry["pages"], meta.get("page", 1), meta.get("total_pages", 1))

        if self.supabase is not None:
            try:
                resp = self.supabase.table("documents").select("metadata").limit(2000).execute()
                sb_counts: Dict[str, int] = {}
                for row in resp.data or []:
                    meta = row.get("metadata") or {}
                    source = meta.get("source")
                    if not source:
                        continue
                    sb_counts[source] = sb_counts.get(source, 0) + 1
                    if source not in docs_map:
                        docs_map[source] = {
                            "filename": source,
                            "chunks": 0,
                            "pages": meta.get("total_pages", meta.get("page", 1)),
                            "uploaded_at": meta.get("uploaded_at"),
                            "has_file": (UPLOADS_DIR / source).exists(),
                        }
                for source, count in sb_counts.items():
                    docs_map[source]["chunks"] = max(docs_map[source]["chunks"], count)
                self.last_storage_backend = "supabase"
            except Exception:
                pass

        return sorted(
            docs_map.values(),
            key=lambda d: d.get("uploaded_at") or "",
            reverse=True,
        )

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """
        Deletes a document's chunks from both local store and Supabase, plus its PDF file.
        """
        records = self._load_local_store()
        remaining = [r for r in records if r.get("metadata", {}).get("source") != filename]
        removed_local = len(records) - len(remaining)
        self._save_local_store(remaining)

        removed_supabase = False
        if self.supabase is not None:
            try:
                self.supabase.table("documents").delete().contains(
                    "metadata", {"source": filename}
                ).execute()
                removed_supabase = True
            except Exception as e:
                print(f"[IngestionService] Supabase delete warning: {e}")

        pdf_path = UPLOADS_DIR / filename
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception:
                pass

        return {
            "filename": filename,
            "removed_chunks": removed_local,
            "removed_from_supabase": removed_supabase,
        }
