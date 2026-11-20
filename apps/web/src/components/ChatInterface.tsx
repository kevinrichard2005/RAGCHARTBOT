"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  FileText,
  User,
  Bot,
  Paperclip,
  Maximize2,
  Minimize2,
  Loader2,
  Trash2,
  Upload,
  ChevronLeft,
  ChevronRight,
  Database,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Filter,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Citation {
  text: string;
  page: number;
  source?: string;
  score?: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  isError?: boolean;
}

interface DocumentInfo {
  filename: string;
  chunks: number;
  pages: number;
  uploaded_at?: string;
  has_file?: boolean;
}

interface HealthInfo {
  status: string;
  documents_count: number;
  total_chunks: number;
  storage_backend: string;
  embedding_mode: string;
  config?: {
    supabase_url_configured: boolean;
    supabase_service_key_looks_publishable: boolean;
    google_api_key_configured: boolean;
    google_api_key_looks_valid: boolean;
  };
}

const INITIAL_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Hello! Upload a PDF using the paperclip icon or the left panel, then ask me anything about your documents. Click any page citation badge to jump directly to that page in the PDF viewer.",
};

const QUICK_PROMPTS = [
  "Summarize the main points of this document",
  "What are the key takeaways and conclusions?",
  "List the important numbers, dates, or metrics mentioned",
];

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [activeDoc, setActiveDoc] = useState<string | null>(null);
  const [activePage, setActivePage] = useState<number>(1);
  const [filterByActiveDoc, setFilterByActiveDoc] = useState<boolean>(false);
  const [localBlobUrls, setLocalBlobUrls] = useState<Record<string, string>>({});
  const [isPdfMaximized, setIsPdfMaximized] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean>(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchBackendState = useCallback(async () => {
    try {
      const [docsRes, healthRes] = await Promise.all([
        fetch(`${API_URL}/documents`),
        fetch(`${API_URL}/health`),
      ]);

      if (docsRes.ok) {
        const docsData = await docsRes.json();
        const docsList: DocumentInfo[] = docsData.documents || [];
        setDocuments(docsList);
        setActiveDoc((prev) => {
          if (prev && docsList.some((d) => d.filename === prev)) return prev;
          return docsList.length > 0 ? docsList[0].filename : null;
        });
      }

      if (healthRes.ok) {
        const healthData: HealthInfo = await healthRes.json();
        setHealth(healthData);
        setBackendOnline(true);
      }
    } catch {
      setBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    fetchBackendState();
  }, [fetchBackendState]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  const currentDocInfo = documents.find((d) => d.filename === activeDoc) || null;
  const maxPages = currentDocInfo?.pages || 1;

  const getPdfViewerUrl = () => {
    if (!activeDoc) return null;
    const baseUrl =
      localBlobUrls[activeDoc] ||
      `${API_URL}/documents/${encodeURIComponent(activeDoc)}/file`;
    return `${baseUrl}#page=${activePage}&view=FitH`;
  };

  const handleCitationClick = (cit: Citation) => {
    if (cit.source && documents.some((d) => d.filename === cit.source)) {
      setActiveDoc(cit.source);
    }
    setActivePage(cit.page || 1);
  };

  const handleSend = async (customPrompt?: string) => {
    const question = (customPrompt ?? input).trim();
    if (!question || isSending) return;

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
    };

    const priorHistory = messages
      .filter((m) => m.id !== "welcome" && !m.isError)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMsg]);
    if (!customPrompt) setInput("");
    setIsSending(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: question,
          history: priorHistory,
          document: filterByActiveDoc && activeDoc ? activeDoc : null,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Server returned ${response.status}`);
      }

      const aiMsg: Message = {
        id: `ai-${Date.now()}`,
        role: "assistant",
        content: data.answer || "No answer returned.",
        citations: data.citations || [],
      };
      setMessages((prev) => [...prev, aiMsg]);
      setBackendOnline(true);
    } catch (error: unknown) {
      const errMsg =
        error instanceof Error
          ? error.message
          : "Could not connect to the FastAPI backend.";
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "assistant",
          content: `Error: ${errMsg}`,
          isError: true,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus("Only PDF files (.pdf) are supported.");
      return;
    }

    // Create instant local blob URL for immediate PDF preview
    const blobUrl = URL.createObjectURL(file);
    setIsUploading(true);
    setUploadStatus(`Uploading & indexing "${file.name}"...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to process PDF.");
      }

      const cleanFilename = data.filename || file.name;
      setLocalBlobUrls((prev) => ({ ...prev, [cleanFilename]: blobUrl }));
      setActiveDoc(cleanFilename);
      setActivePage(1);
      setUploadStatus(null);

      await fetchBackendState();

      setMessages((prev) => [
        ...prev,
        {
          id: `upload-${Date.now()}`,
          role: "assistant",
          content: `Indexed **${cleanFilename}** (${data.chunks} chunks across ${data.pages} page${data.pages === 1 ? "" : "s"}). Ask me any question about it!`,
        },
      ]);
    } catch (error: unknown) {
      const errText =
        error instanceof Error ? error.message : "Failed to upload document.";
      setUploadStatus(`Upload failed: ${errText}`);
      setMessages((prev) => [
        ...prev,
        {
          id: `upload-err-${Date.now()}`,
          role: "assistant",
          content: `Failed to index "${file.name}": ${errText}`,
          isError: true,
        },
      ]);
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    try {
      const res = await fetch(
        `${API_URL}/documents/${encodeURIComponent(filename)}`,
        { method: "DELETE" }
      );
      if (res.ok) {
        await fetchBackendState();
      }
    } catch (err) {
      console.error("Delete document failed:", err);
    }
  };

  const pdfViewerUrl = getPdfViewerUrl();

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white overflow-hidden">
      {/* Left Side: Interactive PDF Viewer & Document Manager */}
      <div
        className={`hidden lg:flex flex-col border-r border-white/10 bg-[#0f0f0f] transition-all duration-300 ${
          isPdfMaximized ? "w-3/5" : "w-1/2"
        }`}
      >
        {/* Top Document Bar */}
        <div className="p-3 border-b border-white/10 flex justify-between items-center bg-[#141414] gap-2">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <FileText className="w-5 h-5 text-blue-400 shrink-0" />
            {documents.length > 0 ? (
              <select
                value={activeDoc || ""}
                onChange={(e) => {
                  setActiveDoc(e.target.value);
                  setActivePage(1);
                }}
                className="bg-[#1f1f1f] text-white text-xs font-medium px-2.5 py-1.5 rounded-md border border-white/10 focus:outline-none focus:border-blue-500 truncate max-w-[260px]"
              >
                {documents.map((doc) => (
                  <option key={doc.filename} value={doc.filename}>
                    {doc.filename} ({doc.pages}p • {doc.chunks} chunks)
                  </option>
                ))}
              </select>
            ) : (
              <span className="font-medium text-xs text-white/50 truncate">
                No PDF uploaded yet
              </span>
            )}
          </div>

          {/* Page Navigation & Actions */}
          <div className="flex items-center gap-1.5 shrink-0">
            {activeDoc && (
              <>
                <div className="flex items-center bg-[#1f1f1f] rounded-md border border-white/10 px-1 py-0.5 text-xs">
                  <button
                    onClick={() => setActivePage((p) => Math.max(1, p - 1))}
                    disabled={activePage <= 1}
                    className="p-1 hover:text-blue-400 disabled:opacity-30"
                    title="Previous Page"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                  </button>
                  <span className="px-2 text-[11px] font-mono text-white/80">
                    p.{activePage}/{maxPages}
                  </span>
                  <button
                    onClick={() =>
                      setActivePage((p) => Math.min(maxPages, p + 1))
                    }
                    disabled={activePage >= maxPages}
                    className="p-1 hover:text-blue-400 disabled:opacity-30"
                    title="Next Page"
                  >
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleDeleteDocument(activeDoc)}
                  className="h-8 w-8 text-white/50 hover:text-red-400 hover:bg-red-500/10"
                  title="Delete Document"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </>
            )}

            <Button
              variant="ghost"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="h-8 px-2.5 text-xs bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 gap-1.5"
            >
              {isUploading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Upload className="w-3.5 h-3.5" />
              )}
              Upload PDF
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsPdfMaximized((v) => !v)}
              className="h-8 w-8 hover:bg-white/5 text-white/60"
              title={isPdfMaximized ? "Restore width" : "Expand PDF viewer"}
            >
              {isPdfMaximized ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Upload status banner if active */}
        {uploadStatus && (
          <div className="px-4 py-2 bg-blue-500/10 border-b border-blue-500/20 flex items-center gap-2 text-xs text-blue-300">
            {isUploading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
            ) : (
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            )}
            <span className="truncate">{uploadStatus}</span>
          </div>
        )}

        {/* PDF Viewer Body */}
        <div className="flex-1 flex items-center justify-center p-4 overflow-hidden">
          {activeDoc && pdfViewerUrl ? (
            <iframe
              key={`${activeDoc}-${activePage}`}
              src={pdfViewerUrl}
              title={`PDF Viewer - ${activeDoc}`}
              className="w-full h-full rounded-lg border border-white/10 bg-[#1a1a1a]"
            />
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="w-full h-full bg-white/[0.02] hover:bg-white/[0.04] transition-colors cursor-pointer rounded-xl border border-dashed border-white/15 flex flex-col items-center justify-center gap-4 p-8 text-center"
            >
              <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                <Upload className="w-7 h-7 text-blue-400" />
              </div>
              <div className="space-y-1 max-w-sm">
                <h3 className="text-sm font-semibold text-white/90">
                  Upload a PDF Document to Begin
                </h3>
                <p className="text-xs text-white/40 leading-relaxed">
                  Click anywhere here or use the paperclip icon in the chat box.
                  Your PDF will be chunked, embedded, and displayed here with
                  interactive page citations.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right Side: Chat Interface */}
      <div className="flex-1 flex flex-col relative min-w-0">
        {/* Header */}
        <div className="p-3.5 border-b border-white/10 flex items-center justify-between gap-3 bg-[#0a0a0a]/90 backdrop-blur-md sticky top-0 z-10">
          <div className="flex items-center gap-3">
            <Avatar className="w-9 h-9 border border-blue-500/30">
              <AvatarImage src="/ai-avatar.png" />
              <AvatarFallback className="bg-blue-600/20 text-blue-400 text-xs font-bold">
                AI
              </AvatarFallback>
            </Avatar>
            <div>
              <h2 className="text-sm font-semibold flex items-center gap-2">
                Document Assistant
                <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-white/60 flex items-center gap-1">
                  <Database className="w-2.5 h-2.5 text-blue-400" />
                  {health?.storage_backend === "supabase"
                    ? "Supabase pgvector"
                    : "Local Vector Store"}
                </span>
              </h2>
              <div className="flex items-center gap-2 mt-0.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    backendOnline
                      ? "bg-green-500 animate-pulse"
                      : "bg-red-500"
                  }`}
                />
                <span className="text-[11px] text-white/50">
                  {backendOnline
                    ? `${documents.length} doc${documents.length === 1 ? "" : "s"} (${health?.total_chunks ?? 0} chunks)`
                    : "Backend Offline (port 8000)"}
                </span>
              </div>
            </div>
          </div>

          {/* Right Header Controls: Scope Filter & Reset */}
          <div className="flex items-center gap-2">
            {activeDoc && (
              <button
                onClick={() => setFilterByActiveDoc((v) => !v)}
                className={`text-xs px-2.5 py-1.5 rounded-lg border transition-colors flex items-center gap-1.5 ${
                  filterByActiveDoc
                    ? "bg-blue-600/20 border-blue-500/40 text-blue-300"
                    : "bg-white/5 border-white/10 text-white/60 hover:text-white"
                }`}
                title="Toggle searching only the currently selected PDF vs all uploaded PDFs"
              >
                <Filter className="w-3 h-3" />
                {filterByActiveDoc ? `Scope: ${activeDoc}` : "Scope: All Docs"}
              </button>
            )}

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMessages([INITIAL_MESSAGE])}
              className="h-8 w-8 text-white/40 hover:text-white hover:bg-white/5"
              title="Clear Chat History"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Config Notice if Gemini API key in .env doesn't look like a standard AIza key */}
        {health?.config && !health.config.google_api_key_looks_valid && (
          <div className="px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 flex items-center justify-between gap-2 text-[11px] text-amber-200/90">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span>
                <strong>Tip:</strong> Your <code>GOOGLE_API_KEY</code> in{" "}
                <code>.env</code> doesn&apos;t start with <code>AIza...</code>.
                Retrieval works via built-in local fallback, and adding a Google
                AI Studio key unlocks full Gemini generative synthesis.
              </span>
            </div>
          </div>
        )}

        {/* Messages Scroll Area */}
        <ScrollArea className="flex-1 p-4 lg:p-6">
          <div className="max-w-2xl mx-auto space-y-6">
            {/* Quick Starter Prompts when only welcome message is present */}
            {messages.length === 1 && documents.length > 0 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {QUICK_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => handleSend(prompt)}
                    className="text-xs bg-white/[0.04] hover:bg-blue-500/15 hover:border-blue-500/30 border border-white/10 text-white/75 hover:text-blue-200 px-3 py-1.5 rounded-full transition-colors flex items-center gap-1.5"
                  >
                    <Sparkles className="w-3 h-3 text-blue-400" />
                    {prompt}
                  </button>
                ))}
              </div>
            )}

            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-3.5 ${
                    msg.role === "user" ? "flex-row-reverse" : ""
                  }`}
                >
                  <Avatar
                    className={`w-8 h-8 shrink-0 ${
                      msg.role === "assistant"
                        ? "border border-blue-500/30"
                        : "border border-white/20"
                    }`}
                  >
                    <AvatarFallback
                      className={
                        msg.role === "assistant"
                          ? "bg-blue-900/50"
                          : "bg-white/10"
                      }
                    >
                      {msg.role === "assistant" ? (
                        <Bot className="w-4 h-4 text-blue-400" />
                      ) : (
                        <User className="w-4 h-4" />
                      )}
                    </AvatarFallback>
                  </Avatar>

                  <div
                    className={`flex flex-col gap-2 max-w-[82%] ${
                      msg.role === "user" ? "items-end" : ""
                    }`}
                  >
                    <div
                      className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap ${
                        msg.isError
                          ? "bg-red-950/50 border border-red-500/30 text-red-200"
                          : msg.role === "assistant"
                          ? "bg-[#1a1a1a] border border-white/5 text-white/90"
                          : "bg-blue-600 text-white"
                      }`}
                    >
                      {msg.content}
                    </div>

                    {msg.citations && msg.citations.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-0.5">
                        {msg.citations.map((cit, idx) => (
                          <TooltipProvider key={idx} delayDuration={150}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <button
                                  onClick={() => handleCitationClick(cit)}
                                  className={`text-[11px] px-2.5 py-1 rounded-md transition-colors flex items-center gap-1.5 border ${
                                    activeDoc === cit.source &&
                                    activePage === cit.page
                                      ? "bg-blue-600/25 border-blue-500/50 text-blue-200"
                                      : "bg-white/5 hover:bg-white/10 border-white/10 text-white/80"
                                  }`}
                                >
                                  <FileText className="w-3 h-3 text-blue-400" />
                                  <span>
                                    {cit.source ? `${cit.source} • ` : ""}p.
                                    {cit.page}
                                  </span>
                                </button>
                              </TooltipTrigger>
                              <TooltipContent className="bg-[#141414] border-white/15 text-white/90 text-xs max-w-sm p-3 leading-relaxed shadow-xl">
                                <p className="font-semibold text-blue-400 mb-1">
                                  {cit.source || "Document"} — Page {cit.page}
                                </p>
                                <p className="text-white/75 italic">
                                  &ldquo;{cit.text}&rdquo;
                                </p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Thinking / Loading Indicator */}
            {isSending && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3.5"
              >
                <Avatar className="w-8 h-8 border border-blue-500/30">
                  <AvatarFallback className="bg-blue-900/50">
                    <Bot className="w-4 h-4 text-blue-400" />
                  </AvatarFallback>
                </Avatar>
                <div className="px-4 py-3 rounded-2xl bg-[#1a1a1a] border border-white/5 text-white/70 text-xs flex items-center gap-2.5">
                  <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  <span>Searching chunks & generating answer...</span>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="p-4 lg:p-5 border-t border-white/10 bg-[#0a0a0a]">
          <div className="max-w-2xl mx-auto relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur opacity-20 group-focus-within:opacity-40 transition duration-500" />
            <div className="relative bg-[#1a1a1a] rounded-xl border border-white/10 p-2 flex items-center gap-2 shadow-2xl">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleUpload}
                className="hidden"
              />
              <Button
                variant="ghost"
                size="icon"
                disabled={isUploading}
                onClick={() => fileInputRef.current?.click()}
                className="text-white/50 hover:text-white hover:bg-white/5 shrink-0"
                title="Upload PDF Document"
              >
                {isUploading ? (
                  <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
                ) : (
                  <Paperclip className="w-5 h-5" />
                )}
              </Button>

              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={isSending}
                placeholder={
                  activeDoc
                    ? `Ask anything about ${
                        filterByActiveDoc ? activeDoc : "your documents"
                      }...`
                    : "Upload a PDF or ask a question..."
                }
                className="bg-transparent border-none focus-visible:ring-0 placeholder:text-white/30 text-sm h-11"
              />

              <Button
                onClick={() => handleSend()}
                disabled={isSending || !input.trim()}
                size="icon"
                className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white shrink-0 shadow-lg shadow-blue-500/20"
              >
                {isSending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>
            <div className="flex items-center justify-between text-[10px] text-white/30 mt-2.5 px-1">
              <span>Press Enter to send • Click any citation to jump to page</span>
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-green-500/70" />
                RAG Pipeline Active
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
