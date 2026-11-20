-- =============================================================================
-- RAG Chatbot — Supabase pgvector Schema & Similarity Search RPC
-- Run this script in your Supabase SQL Editor (https://supabase.com/dashboard)
-- =============================================================================

-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Create the documents table (768 dimensions for Google Gemini embeddings)
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(768),
  created_at timestamptz not null default now()
);

-- 3. Create indexes for fast vector similarity search and metadata filtering
create index if not exists documents_embedding_hnsw_idx
  on public.documents
  using hnsw (embedding vector_cosine_ops);

create index if not exists documents_metadata_gin_idx
  on public.documents
  using gin (metadata jsonb_path_ops);

-- 4. Enable Row Level Security (RLS)
alter table public.documents enable row level security;

-- Allow read/write access (adjust to `to authenticated` if adding user auth later)
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'documents' and policyname = 'Allow public read access to documents'
  ) then
    create policy "Allow public read access to documents"
      on public.documents
      for select
      to anon, authenticated
      using (true);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'documents' and policyname = 'Allow public insert access to documents'
  ) then
    create policy "Allow public insert access to documents"
      on public.documents
      for insert
      to anon, authenticated
      with check (true);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'documents' and policyname = 'Allow public delete access to documents'
  ) then
    create policy "Allow public delete access to documents"
      on public.documents
      for delete
      to anon, authenticated
      using (true);
  end if;
end
$$;

-- 5. Similarity search RPC function (compatible with LangChain & direct PostgREST RPC)
create or replace function public.match_documents (
  query_embedding vector(768),
  match_count int default 4,
  filter jsonb default '{}'::jsonb
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
stable
security invoker
as $$
#variable_conflict use_column
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from public.documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
