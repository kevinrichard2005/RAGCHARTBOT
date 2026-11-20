# RAG Chatbot Monorepo

Next-generation AI Document Search & Question-Answering engine with interactive page citations.

## 🚀 Quick Start

### 1. One-Command Run (Windows PowerShell)
```powershell
.\start.ps1
```
This launches:
- **FastAPI Backend**: `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`, diagnostics at `http://localhost:8000/health`)
- **Next.js Frontend**: `http://localhost:3000`

### 2. Supabase `pgvector` Setup (Optional — Local Vector Store Fallback Built-In)
The backend works out-of-the-box using a persistent **Local Vector Store** (`apps/api/data/vector_store.json`) and automatically syncs with **Supabase `pgvector`** when configured.

To enable Supabase `pgvector`:
1. Open your Supabase project's **SQL Editor**.
2. Paste and run [`supabase/schema.sql`](./supabase/schema.sql) (creates the `documents` table, HNSW vector index, RLS policies, and `match_documents` RPC function).

### 3. Environment Setup (`.env`)
Configure `.env` in the root directory:
```env
# Backend (Get a Gemini API key starting with AIza... from https://aistudio.google.com/apikey)
GOOGLE_API_KEY=your_gemini_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_role_or_anon_key

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Manual Run

#### Backend (FastAPI)
```powershell
cd apps/api
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend (Next.js)
```powershell
cd apps/web
npm run dev
```

## ✨ Features
- **PDF Ingestion & Deduplication**: Extracts text & page metadata with `PyPDFLoader`, splits into semantic chunks (`1000` chars, `150` overlap), deduplicates re-uploads, and serves the PDF inline.
- **Dual Vector Storage**: Supports **Supabase `pgvector`** (`match_documents` RPC) with automatic fallback to a persistent **Local Vector Store** + hybrid keyword re-ranking.
- **Conversational RAG**: Uses chat history to contextualize follow-up questions and supports scoping queries to a single PDF or across all uploaded documents.
- **Interactive PDF Viewer & Page Citations**: Click any `filename • p.X` citation badge under an answer to jump directly to page `X` in the side-by-side PDF viewer.
