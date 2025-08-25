We’ll use ChromaDB (local vector DB) for simplicity, openai embeddings, and your FastAPI agent.

Folder Structure (Updated for RAG)
floki_agent/
│
├─ .venv/
├─ .env
├─ .gitignore
├─ .python-version
├─ api.py     #FastAPI + vectorstore endpoints
├─ config.py
├─ db.py         #database models & session
├─ floki_agent.py      #AI agent logic (embeddings, summarizer, user info)
├─ pyproject.toml
├─ README.md
├─ requirements.txt
|
└─ uv.lock



===============================------------------------------------------------------------------------------


1️⃣ .env

# Gemini API Key
GEMINI_API_KEY=""

# Database URL (Postgres / SQLite / Supabase)
DATABASE_URL=postgresql+asyncpg://postgres:admin@localhost/floki

# Vector DB path (for Chroma)
VECTOR_DB_PATH=./vector_db




==============================---------------------------------------------------------------------

2️⃣ config.py
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
==============================---------------------------------------------------------------------
3️⃣ db.py (same as before)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from datetime import datetime
from config import DATABASE_URL

# --------------------------
# Base & Engine
# --------------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)  # PostgreSQL async engine
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# --------------------------
# Users
# --------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    age = Column(Integer, nullable=True)
    location = Column(String, nullable=True)
    funding_status = Column(String, nullable=True)  # demo, funded, etc.
    account_type = Column(String, nullable=True)    # FTMO, MFF, Apex, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    trades = relationship("Trade", back_populates="user")
    emotions = relationship("Emotion", back_populates="user")
    journals = relationship("Journal", back_populates="user")
    feature_usage = relationship("FeatureUsage", back_populates="user")
    reset_challenges = relationship("ResetChallenge", back_populates="user")
    recovery_plans = relationship("RecoveryPlan", back_populates="user")
    rulebook_votes = relationship("RulebookVote", back_populates="user")
    simulator_logs = relationship("SimulatorLog", back_populates="user")

# --------------------------
# Trades
# --------------------------
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    instrument = Column(String)
    strategy = Column(String)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    outcome = Column(String)  # win/loss
    r_r_ratio = Column(Float)
    max_drawdown = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="trades")
    emotions = relationship("Emotion", back_populates="trade")

# --------------------------
# Emotions
# --------------------------
class Emotion(Base):
    __tablename__ = "emotions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    emotion_tag = Column(String)        # fear, confidence, anger
    confidence_level = Column(Float)    # 0-100
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="emotions")
    trade = relationship("Trade", back_populates="emotions")

# --------------------------
# Journals
# --------------------------
class Journal(Base):
    __tablename__ = "journals"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(Text)
    sentiment_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="journals")

# --------------------------
# Feature Usage
# --------------------------
class FeatureUsage(Base):
    __tablename__ = "feature_usage"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    feature_name = Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feature_usage")


# --------------------------
# Reset Challenges
# --------------------------
class ResetChallenge(Base):
    __tablename__ = "reset_challenges"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    challenge_name = Column(String)
    completed = Column(Boolean, default=False)
    progress_percent = Column(Float, default=0.0)
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    user = relationship("User", back_populates="reset_challenges")

# --------------------------
# Recovery Plans
# --------------------------
class RecoveryPlan(Base):
    __tablename__ = "recovery_plans"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_name = Column(String)
    completed = Column(Boolean, default=False)
    progress_percent = Column(Float, default=0.0)
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    user = relationship("User", back_populates="recovery_plans")

# --------------------------
# Rulebook Votes
# --------------------------
class RulebookVote(Base):
    __tablename__ = "rulebook_votes"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    rule_name = Column(String)
    vote = Column(String)  # yes/no/abstain
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="rulebook_votes")

# --------------------------
# Simulator Logs
# --------------------------
class SimulatorLog(Base):
    __tablename__ = "simulator_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="simulator_logs")

# --------------------------
# News Items
# --------------------------
# --------------------------
# News Items
# --------------------------
class NewsItem(Base):
    __tablename__ = "news_items"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    url = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    detailed_summary = Column(Text, nullable=True)
    text = Column(Text, nullable=True)  # ✅ Added this column
    keywords = Column(String, nullable=True)
    keys = Column(String, nullable=True)  # ✅ Add this column


# --------------------------
# Create All Tables Helper
# --------------------------
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ All tables created successfully!")

====================================================---------------------------------------------------------------------
==============================---------------------------------------------------------------------
04 floki_agent_gemini.py
import sys
import os
import asyncio
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime

import chromadb
from sentence_transformers import SentenceTransformer
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from openai import AsyncOpenAI
from config import DATABASE_URL, load_dotenv

# -----------------------------
# Windows asyncio fix
# -----------------------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -----------------------------
# Disable tracing for cleaner logs
# -----------------------------
set_tracing_disabled(True)

# -----------------------------
# Database setup
# -----------------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class SessionQuery(Base):
    __tablename__ = "session_queries"
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    query_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ All tables created successfully!")

# -----------------------------
# ChromaDB setup
# -----------------------------
client_chroma = chromadb.Client()
collection = client_chroma.get_or_create_collection("floki_embeddings")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def add_to_vectorstore(text: str, metadata: dict = None):
    vector = embedding_model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[vector],
        ids=[str(hash(text))],
        metadatas=[metadata or {}]
    )

def search_vectorstore(query: str, n_results: int = 3):
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

# -----------------------------
# Load OpenAI Gemini
# -----------------------------
load_dotenv()
client_ai = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model_ai = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client_ai,
)

# -----------------------------
# Floki instructions
# -----------------------------
FLOKI_INSTRUCTION = (
    "You are Floki, a friendly AI assistant for FundedFlow premium users. "
    "Always greet users warmly with a short intro like 'Hi! I’m Floki 👋 Your trading buddy.' "
    "Do NOT give long context or advice until the user asks or engages. "
    "Provide full module explanations only when requested. "
    "Focus on helping the current user improve their trading skills and mindset. "
    "Do NOT share other users’ personal info under any circumstances. "
    "Follow all FundedFlow rules and privacy policies. "
    "Politely say you can only answer trading-related questions if unrelated. "
    "Use emojis and short, friendly sentences!"
)

# -----------------------------
# Database helpers
# -----------------------------
async def save_query(session_id: str, query_text: str):
    async with async_session() as session:
        sq = SessionQuery(session_id=session_id, query_text=query_text)
        session.add(sq)
        await session.commit()

async def get_past_queries(session_id: str):
    async with async_session() as session:
        result = await session.execute(
            SessionQuery.__table__.select().where(SessionQuery.session_id == session_id)
        )
        rows = result.fetchall()
        return [r.query_text for r in rows]

# -----------------------------
# Agent tools
# -----------------------------
async def summarize_text(text: str) -> str:
    resp = await client_ai.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": f"Summarize this text in 3 bullet points:\n{text}"}],
        temperature=0.3
    )
    return resp.choices[0].message.content

async def get_user_info(user_id: str) -> str:
    return "You're using FundedFlow premium! Let's focus on improving your trades and mindset. 🚀"

async def get_user_history(user_id: str) -> str:
    results = search_vectorstore("trading", n_results=3)
    try:
        docs = results.get('documents', [])
        if docs and isinstance(docs[0], list):
            return " | ".join(docs[0])
        elif docs:
            return " | ".join([str(d) for d in docs])
        else:
            return "No trading history yet."
    except Exception as e:
        return "Error fetching history."

async def get_user_relevant_news(user_id: str) -> str:
    results = search_vectorstore("stocks", n_results=3)
    return " | ".join(results['documents'][0]) if results['documents'] else "No news found."

async def suggest_improvements(session_id: str) -> str:
    queries = await get_past_queries(session_id)
    if not queries:
        return "No past session data to suggest improvements yet. 🚀"
    text = "\n".join(queries)
    resp = await client_ai.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": f"Provide trading improvement tips based on these past queries:\n{text}"}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# -----------------------------
# Register tools
# -----------------------------
summarize_tool = function_tool(summarize_text)
get_user_info_tool = function_tool(get_user_info)
get_user_history_tool = function_tool(get_user_history)
get_user_news_tool = function_tool(get_user_relevant_news)
suggest_improvements_tool = function_tool(suggest_improvements)

# -----------------------------
# Floki Agent
# -----------------------------
floki_agent = Agent(
    name="FlokiMentor",
    instructions=FLOKI_INSTRUCTION,
    tools=[
        summarize_tool,
        get_user_info_tool,
        get_user_history_tool,
        get_user_news_tool,
        suggest_improvements_tool
    ],
    model=model_ai
)

# -----------------------------
# Website fetch & embed
# -----------------------------
import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timezone

async def fetch_and_embed_website(url: str):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]

            for text in paragraphs:
                unique_id = str(hash(url + text))
                add_to_vectorstore(text, {"source": url, "timestamp": datetime.now(timezone.utc).isoformat(), "id": unique_id})

        print(f"✅ Embedded {len(paragraphs)} paragraphs from {url}")

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")

# -----------------------------
# Interactive loop with dynamic intro
# -----------------------------
async def interactive_loop():
    session_id = "current_session"
    print("🟢 Floki Agent Interactive Testing Started! Type 'exit' to quit.\n")

    # Generate intro dynamically
    intro = await Runner.run(floki_agent, "Start with a short friendly intro message for a premium user.", context={"user_id": session_id})
    print(f"💬 Floki: {intro.final_output}\n")

    while True:
        query = input("Enter your query: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("🛑 Exiting Floki Agent testing loop.")
            break

        # Save query first (so we have context)
        await save_query(session_id, query)
        add_to_vectorstore(query, {
            "user_id": session_id,
            "source": "interactive_loop",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        print("✅ Query saved to DB and added to vectorstore.\n")

        # Detect if user wants advice
        if any(word in query.lower() for word in ["help", "advise", "tips", "improve", "strategy"]):
            # User asked for guidance → give improvements
            tips = await suggest_improvements(session_id)
            print("💡 Floki Improvement Tips:\n", tips, "\n")
        else:
            # Just listen first
            response = await Runner.run(floki_agent, query, context={"user_id": session_id})
            print("\n💬 Floki Agent Response:\n", response.final_output)
            print("🟢 Floki is listening. Say 'help' or 'tips' if you want improvement advice.\n")


# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    async def main():
        await init_db()
        await interactive_loop()

    asyncio.run(main())



============================---------------------------------------------------------------------=======================
============================================--------------------------------------------------------------vvvv------------------------------------------------------------------------------------------
05 api.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from floki_agent import (
    Runner,
    floki_agent,
    search_vectorstore,
    get_user_info,
    get_user_history,
    get_user_relevant_news,
    save_query,
    add_to_vectorstore,
    fetch_and_embed_website,
    get_past_queries,
    collection
)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Floki Live RAG Agent")

# -----------------------------
# Request Models
# -----------------------------
class QueryRequest(BaseModel):
    user_id: str
    query: str
    n_results: int = 3

class EmbedRequest(BaseModel):
    url: str

# -----------------------------
# Helper functions
# -----------------------------
def flatten_documents(docs):
    """Flatten nested list of documents safely"""
    flat = []
    for d in docs:
        if isinstance(d, list):
            flat.extend([str(x) for x in d])
        else:
            flat.append(str(d))
    return flat

# -----------------------------
# GET endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Floki Live RAG Agent Running!"}

@app.get("/past_queries")
async def past_queries(user_id: str = Query(..., description="User ID to fetch past queries")):
    try:
        queries = await get_past_queries(user_id)
        return {"user_id": user_id, "queries": queries}
    except Exception as e:
        logger.exception(f"Error fetching past queries for {user_id}")
        raise HTTPException(status_code=500, detail="Failed to fetch past queries.")

@app.get("/vectorstore_status")
async def vectorstore_status():
    try:
        num_docs = len(collection.get(include=["documents"]).get("documents", []))
        return {"num_documents": num_docs}
    except Exception as e:
        logger.exception("Error fetching vectorstore status")
        raise HTTPException(status_code=500, detail="Failed to fetch vectorstore status.")

# -----------------------------
# POST endpoints
# -----------------------------
@app.post("/ask")
async def ask_floki(request: QueryRequest):
    try:
        # 1️⃣ Search vectorstore for relevant context
        results = search_vectorstore(request.query, n_results=request.n_results)
        flat_docs = flatten_documents(results.get("documents", []))
        relevant_texts = " | ".join(flat_docs) if flat_docs else "No relevant info found."

        # 2️⃣ Get user info, history, and news
        user_info = await get_user_info(request.user_id)
        user_history = await get_user_history(request.user_id)
        user_news = await get_user_relevant_news(request.user_id)

        # 3️⃣ Build full prompt
        full_prompt = (
            f"{user_info}\nHistory: {user_history}\nRelevant news: {user_news}\n"
            f"Vectorstore context: {relevant_texts}\nUser Query: {request.query}"
        )

        # 4️⃣ Generate response using Runner
        response = await Runner.run(floki_agent, full_prompt, context={"user_id": request.user_id})

        # 5️⃣ Save query in DB and vectorstore
        await save_query(request.user_id, request.query)
        add_to_vectorstore(request.query, {"user_id": request.user_id, "source": "ask_endpoint"})

        return {
            "response": getattr(response, "final_output", str(response)),
            "vectorstore_context": relevant_texts,
            "user_history": user_history
        }

    except Exception as e:
        logger.exception("Error in /ask endpoint")
        raise HTTPException(status_code=500, detail="Failed to process request.")
#=============================================

#===============================================
@app.post("/embed_website")
async def embed_website(request: EmbedRequest):
    # ✅ Validate URL first
    if not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    
    try:
        await fetch_and_embed_website(request.url)
        return {"message": f"✅ Website {request.url} fetched and embedded successfully."}
    except Exception as e:
        logger.exception(f"Error embedding website {request.url}")
        raise HTTPException(status_code=500, detail="Failed to fetch and embed website.")



# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki Live RAG Agent starting up...")
    try:
        await fetch_and_embed_website("https://fundedflow.app/")
        logger.info("✅ FundedFlow website embedded successfully at startup.")
    except Exception as e:
        logger.exception("❌ Failed to embed FundedFlow website at startup")

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


==================================================================================================================================


uvicorn api:app --reload

==
llmfor the embeddig 


pip install requests beautifulsoup4
pip install sentence-transformers
pip install chromadb
