import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase.client import Client, create_client

from app.services.ingestion import IngestionService

_API_DIR = Path(__file__).resolve().parents[2]
_ROOT_DIR = _API_DIR.parent.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_API_DIR / ".env")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_overlap_score(query: str, content: str) -> float:
    q_tokens = set(re.findall(r"[a-z0-9_]{2,}", query.lower()))
    if not q_tokens:
        return 0.0
    c_tokens = set(re.findall(r"[a-z0-9_]{2,}", content.lower()))
    overlap = len(q_tokens & c_tokens)
    return overlap / len(q_tokens)


class RetrievalService:
    """
    Handles the Retrieval + Generation half of RAG:
      1. Contextualize follow-up questions with conversation history
      2. Embed the question (Gemini embeddings with local fallback)
      3. Retrieve top-k chunks from Supabase pgvector (with local store fallback + keyword hybrid boost)
      4. Generate a cited answer with Gemini (or grounded extractive fallback if API key is invalid)
    """

    def __init__(self, ingestion_service: Optional[IngestionService] = None):
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

        self.ingestion = ingestion_service or IngestionService()

        self.supabase: Optional[Client] = None
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
            except Exception as e:
                print(f"[RetrievalService] Supabase init warning: {e}")

        self.llm_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    def _build_search_query(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Enriches short follow-up questions (e.g., 'what about page 2?' or 'tell me more about that')
        with recent user turns so vector retrieval stays on-topic.
        """
        if not history:
            return question

        recent_user_msgs = [
            m.get("content", "").strip()
            for m in history[-4:]
            if m.get("role") == "user" and m.get("content", "").strip() and m.get("content", "").strip() != question.strip()
        ]
        if not recent_user_msgs:
            return question

        # If the question is short or contains pronouns, append the previous user question context
        if len(question.split()) <= 7 or re.search(r"\b(it|its|that|this|they|their|those|these|more|also)\b", question.lower()):
            return f"{recent_user_msgs[-1]} — {question}"
        return question

    def _retrieve_chunks(
        self,
        search_query: str,
        k: int = 4,
        document_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding, _ = self.ingestion.embed_query(search_query)
        results: List[Dict[str, Any]] = []

        # 1. Try Supabase pgvector match_documents RPC first
        if self.supabase is not None:
            try:
                rpc_filter = {"source": document_filter} if document_filter else {}
                resp = self.supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_count": k,
                        "filter": rpc_filter,
                    },
                ).execute()
                if resp.data:
                    for row in resp.data:
                        meta = row.get("metadata") or {}
                        content = row.get("content") or ""
                        sim = float(row.get("similarity") or 0.0)
                        results.append(
                            {
                                "content": content,
                                "metadata": meta,
                                "score": sim,
                                "backend": "supabase",
                            }
                        )
                    self.ingestion.last_storage_backend = "supabase"
                    return results
            except Exception as e:
                print(f"[RetrievalService] Supabase RPC fallback to local store: {e}")

        # 2. Local vector store + hybrid keyword re-ranking fallback
        local_records = self.ingestion._load_local_store()
        if document_filter:
            local_records = [
                r for r in local_records
                if r.get("metadata", {}).get("source") == document_filter
            ]

        scored: List[Dict[str, Any]] = []
        for rec in local_records:
            content = rec.get("content", "")
            meta = rec.get("metadata", {})
            emb = rec.get("embedding", [])
            vec_score = _cosine_similarity(query_embedding, emb)
            kw_score = _keyword_overlap_score(search_query, content)
            combined_score = 0.7 * vec_score + 0.3 * kw_score
            scored.append(
                {
                    "content": content,
                    "metadata": meta,
                    "score": round(combined_score, 4),
                    "backend": "local",
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]

    def _extractive_fallback_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        llm_error: Optional[str] = None,
    ) -> str:
        """
        Provides a clean, grounded answer directly from the retrieved PDF chunks
        if the Gemini API key in .env is missing or invalid.
        """
        lines = [
            "Here are the most relevant excerpts found in your uploaded documents:\n"
        ]
        for idx, chunk in enumerate(chunks[:3], start=1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "document.pdf")
            page = meta.get("page", 1)
            snippet = " ".join(chunk.get("content", "").split())
            if len(snippet) > 380:
                snippet = snippet[:380].rsplit(" ", 1)[0] + "..."
            lines.append(f"**{idx}. [{source}, Page {page}]**\n> {snippet}\n")

        if llm_error:
            lines.append(
                "\n---\n"
                "*Note: Gemini LLM generation is currently running in extractive fallback mode because "
                "`GOOGLE_API_KEY` could not be authenticated with Google AI Studio (keys typically start with `AIza...`). "
                "Update `GOOGLE_API_KEY` in `.env` for full conversational synthesis.*"
            )
        return "\n".join(lines)

    async def answer_question(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        document_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        search_query = self._build_search_query(question, history)
        chunks = self._retrieve_chunks(search_query, k=4, document_filter=document_filter)

        if not chunks:
            scope_msg = f" in **{document_filter}**" if document_filter else ""
            return {
                "answer": (
                    f"I couldn't find any indexed content{scope_msg}. "
                    "Please upload a PDF document using the paperclip button (or the Upload button in the PDF panel) to start chatting!"
                ),
                "citations": [],
            }

        # Build context block + deduplicated citations
        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            page = int(meta.get("page", 1))
            source = meta.get("source", "document.pdf")
            content = chunk.get("content", "").strip()
            score = float(chunk.get("score", 0.0))

            context_blocks.append(f"[Source: {source}, Page {page}]\n{content}")
            citations.append(
                {
                    "source": source,
                    "page": page,
                    "text": content[:240] + ("..." if len(content) > 240 else ""),
                    "score": round(score, 3),
                }
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        # Format recent conversation history
        history_lines: List[str] = []
        if history:
            for msg in history[-6:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                text = (msg.get("content") or "").strip()
                if text:
                    history_lines.append(f"{role}: {text[:400]}")
        history_block = "\n".join(history_lines) if history_lines else "No previous turns."

        prompt = (
            "You are an expert AI Document Assistant. Answer the user's question accurately "
            "using ONLY the provided document context below.\n"
            "- Always cite the source document and page number inline like [Source: filename.pdf, p.X].\n"
            "- If the answer is not contained in the context, state clearly what is and isn't in the uploaded document.\n\n"
            f"Recent Conversation History:\n{history_block}\n\n"
            f"Document Context:\n{context_str}\n\n"
            f"User Question: {question}\n\n"
            "Answer:"
        )

        # Try Gemini models in priority order
        last_error: Optional[str] = None
        if self.google_api_key:
            for model_name in self.llm_models:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=self.google_api_key,
                        temperature=0.2,
                    )
                    response = await llm.ainvoke(prompt)
                    answer_text = response.content if hasattr(response, "content") else str(response)
                    if isinstance(answer_text, list):
                        answer_text = "\n".join(str(part) for part in answer_text)
                    return {
                        "answer": answer_text,
                        "citations": citations,
                    }
                except Exception as e:
                    last_error = str(e)
                    print(f"[RetrievalService] Gemini model {model_name} failed: {e}")

        # Fallback to grounded extractive summary if Gemini key is invalid/offline
        return {
            "answer": self._extractive_fallback_answer(question, chunks, last_error),
            "citations": citations,
        }
