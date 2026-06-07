# ─────────────────────────────────────────
# NovaDXB — rag_engine.py
# RAG Pipeline — Phase 1 Simple Skeleton
# ─────────────────────────────────────────

import os
from dotenv import load_dotenv

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine

# Pinecone
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

KNOWLEDGE_BASE_DIR  = "data"
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "novadxb")
EMBED_MODEL         = "text-embedding-ada-002"
LLM_MODEL           = "gpt-4o-mini"
CHUNK_SIZE          = 512
CHUNK_OVERLAP       = 64

# ─────────────────────────────────────────
# GLOBAL — query engine (loaded once)
# ─────────────────────────────────────────

query_engine = None


# ─────────────────────────────────────────
# STEP 1 — Connect to Pinecone
# ─────────────────────────────────────────

def get_pinecone_index():
    """Connect to existing Pinecone index."""
    pc    = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    return index


# ─────────────────────────────────────────
# STEP 2 — Configure LlamaIndex Settings
# ─────────────────────────────────────────

def configure_settings():
    """Set global LlamaIndex embedding and LLM settings."""
    Settings.embed_model = OpenAIEmbedding(
        model=EMBED_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    Settings.chunk_size    = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    print(f"✅ LlamaIndex settings configured")


# ─────────────────────────────────────────
# STEP 3 — Ingest Knowledge Base
# ─────────────────────────────────────────

def ingest_knowledge_base(pinecone_index):
    """
    Load all KB files from /data folder,
    chunk them and push embeddings to Pinecone.
    Run this ONCE to populate the index.
    """
    print(f"📂 Loading knowledge base from: {KNOWLEDGE_BASE_DIR}")

    # Load TXT files via SimpleDirectoryReader
    txt_documents = SimpleDirectoryReader(
        input_dir=KNOWLEDGE_BASE_DIR,
        required_exts=[".txt"]
    ).load_data()

    # Load CSV files manually using pandas (avoids comma-in-field errors)
    import pandas as pd
    import glob
    from llama_index.core import Document as LlamaDocument

    csv_documents = []
    for csv_file in glob.glob(f"{KNOWLEDGE_BASE_DIR}/*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding="utf-8", engine="python", on_bad_lines="skip")
            for _, row in df.iterrows():
                text = "\n".join([f"{col}: {val}" for col, val in row.items()])
                csv_documents.append(LlamaDocument(text=text))
            print(f"✅ Loaded CSV: {csv_file} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Skipped {csv_file}: {e}")

    # Combine all documents
    documents = txt_documents + csv_documents

    print(f"📄 Loaded {len(documents)} documents")

    # Chunk documents
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Connect to Pinecone vector store
    vector_store     = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context  = StorageContext.from_defaults(vector_store=vector_store)

    # Build index — embeds and pushes to Pinecone
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True
    )

    print(f"✅ Knowledge base ingested into Pinecone successfully")
    return index


# ─────────────────────────────────────────
# STEP 4 — Load Existing Index
# ─────────────────────────────────────────

def load_index(pinecone_index):
    """
    Load already-ingested index from Pinecone.
    Used on every app startup after first ingest.
    """
    vector_store    = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

    print(f"✅ Index loaded from Pinecone")
    return index


# ─────────────────────────────────────────
# STEP 5 — Build Query Engine
# ─────────────────────────────────────────

def build_query_engine(index):
    """Build retriever query engine from index."""
    from llama_index.core import PromptTemplate

    # System prompt — tells GPT how to use retrieved context
    SYSTEM_PROMPT = (
        "You are NovaDXB, a premium AI concierge for Dubai tourism. "
        "Use the provided context to give specific, helpful answers. "
        "Always mention real area names, restaurant names, or attraction names from the context. "
        "Be friendly, concise and specific. Never say 'mid-budget area' — always use real names. "
        "If asked about areas, mention 2-3 specific Dubai neighborhoods with details. "
        "If asked about food, mention specific restaurant names and dishes. "
        "If asked about activities, mention specific attraction names and costs."
    )

    qa_template = PromptTemplate(
        "Context from NovaDXB knowledge base:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "System: " + SYSTEM_PROMPT + "\n"
        "Question: {query_str}\n"
        "Answer: "
    )

    engine = index.as_query_engine(
        similarity_top_k=5,
        streaming=False,
        text_qa_template=qa_template
    )
    print(f"✅ Query engine ready")
    return engine


# ─────────────────────────────────────────
# STEP 6 — Initialize RAG (called on startup)
# ─────────────────────────────────────────

def initialize_rag():
    """
    Full initialization pipeline.
    Called once on app startup.
    """
    global query_engine

    print("🚀 Initializing NovaDXB RAG engine...")

    configure_settings()
    pinecone_index = get_pinecone_index()

    # TODO: Set INGEST=true in .env to re-ingest KB
    # Normal startup just loads existing index
    if os.environ.get("INGEST", "false").lower() == "true":
        index = ingest_knowledge_base(pinecone_index)
    else:
        index = load_index(pinecone_index)

    query_engine = build_query_engine(index)
    print("✅ RAG engine initialized and ready")


# ─────────────────────────────────────────
# STEP 7 — Query Function (called by app.py)
# ─────────────────────────────────────────

def query_rag(user_message: str) -> str:
    """
    Main function called by app.py /chat endpoint.
    Takes user message, returns RAG response.
    """
    global query_engine

    if query_engine is None:
        return "RAG engine not initialized yet. Please wait."

    try:
        response = query_engine.query(user_message)
        return str(response)

    except Exception as e:
        print(f"RAG query error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


# ─────────────────────────────────────────
# QUICK TEST — run directly to verify
# ─────────────────────────────────────────

if __name__ == "__main__":
    initialize_rag()
    test_query = "What are the best areas to stay in Dubai for a couple?"
    print(f"\n🔍 Test Query: {test_query}")
    print(f"💬 Response: {query_rag(test_query)}")