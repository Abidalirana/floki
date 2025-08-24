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
Perfect! Let’s set up a learning RAG agent using Gemini so you can run it locally and test how it fetches, summarizes, and answers queries. We’ll keep it simple and fully async.

1️⃣ .env

# Gemini API Key
GEMINI_API_KEY="AIzaSyCXavKRPDnovLX5ls6JyNb7urkN8LQKw2M"

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
class NewsItem(Base):
    __tablename__ = "news_items"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    url = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    detailed_summary = Column(Text, nullable=True)
    keywords = Column(String, nullable=True)

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
# floki_agent.py
# floki_agent.py

import asyncio
from sqlalchemy import text
from datetime import datetime, timezone

from vectorstore import fetch_and_embed_all, scrape_and_embed_website, search_vectorstore, add_documents_to_vectorstore
from db import async_session
from config import GEMINI_API_KEY
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from openai import AsyncOpenAI

# -----------------------------
# Disable tracing for cleaner logs
# -----------------------------
set_tracing_disabled(True)

# -----------------------------
# Gemini client (AsyncOpenAI)
# -----------------------------
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -----------------------------
# Floki system instruction
# -----------------------------
FLOKI_INSTRUCTION = (
    "Your name is Floki. You are a friendly, helpful, and super encouraging AI onboarding "
    "and support bot for FundedFlow. FundedFlow is a private dashboard that helps traders "
    "reset their mindset, improve risk habits, and simulate challenges. "
    "You are informal, easy to understand, and write short messages like a friend. Avoid long, boring paragraphs. "
    "Always keep your messages concise and to the point, like a quick chat. "
    "Gently guide traders to the right tool or their next best action. Always encourage, never judge. "
    "Your primary function is to explain FundedFlow's modules, guide users on how to use them, and provide a general overview of FundedFlow itself. "
    "You can also answer general questions about trading concepts, but always try to relate them back to how FundedFlow's tools or principles can help. "
    "For example, if someone asks 'What is risk management?', explain it generally, then add how the Risk Tracker helps with it! "
    "You can only answer questions related to FundedFlow's modules: '7-Day Reset Challenge', 'Risk Tracker', 'Trading Journal', 'Recovery Plan Generator', 'Loyalty Program', and 'Trading Simulator', "
    "or provide a general overview of FundedFlow, AND general trading concepts. "
    "If a user asks about something completely unrelated to trading or FundedFlow, politely state that you can only help with trading-related topics. "
    "Remember to use emojis and exclamation points to sound friendly and enthusiastic!"
)

# -----------------------------
# Model setup
# -----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)

# -----------------------------
# Async helper functions
# -----------------------------
async def summarize_text(text: str) -> str:
    resp = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": f"Summarize this text in 3 bullet points:\n{text}"}],
        temperature=0.3
    )
    return resp.choices[0].message.content

async def get_user_info(user_id: str) -> str:
    return f"User {user_id} | Name: Abid | Role: Reader"

async def get_user_history(user_id: str) -> str:
    async with async_session() as session:
        result = await session.execute(text("SELECT title FROM news_items ORDER BY id DESC LIMIT 5"))
        rows = result.fetchall()
        return " | ".join([row[0] for row in rows]) if rows else "No history yet."

async def get_user_relevant_news(user_id: str) -> str:
    results = search_vectorstore("stocks", n_results=3)
    return " | ".join(results['documents'][0]) if results['documents'] else "No news found."

# -----------------------------
# Register functions as Agent tools
# -----------------------------
summarize_text_tool = function_tool(summarize_text)
get_user_info_tool = function_tool(get_user_info)
get_user_history_tool = function_tool(get_user_history)
get_user_relevant_news_tool = function_tool(get_user_relevant_news)

# -----------------------------
# Floki Agent
# -----------------------------
news_agent = Agent(
    name="FlokiNewsAssistant",
    instructions=FLOKI_INSTRUCTION,
    tools=[
        summarize_text_tool,
        get_user_info_tool,
        get_user_history_tool,
        get_user_relevant_news_tool,
    ],
    model=model
)

# -----------------------------
# Interactive CLI loop
# -----------------------------
async def interactive_loop():
    user_id = "1234"

    print("🟢 Floki Agent Interactive Testing Started! Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() in ["exit", "quit"]:
            print("🛑 Exiting Floki Agent testing loop.")
            break

        # Fetch info/history/news
        user_info = await get_user_info(user_id)
        history = await get_user_history(user_id)
        relevant_news = await get_user_relevant_news(user_id)
        full_prompt = f"{user_info}\nHistory: {history}\nRelevant news: {relevant_news}\nQuery: {query}"

        # Run agent
        summarized_text = await Runner.run(news_agent, full_prompt)
        print("\n💬 Floki Agent Response:\n", summarized_text.final_output)

        # ✅ Add last query to vectorstore (fixed metadata)
        add_documents_to_vectorstore([
            {
                "text": query,
                "metadata": {  # ✅ singular key
                    "user_id": user_id,
                    "source": "interactive_loop",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        ])
        print("✅ Query added to vectorstore.\n")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    asyncio.run(interactive_loop())




============================---------------------------------------------------------------------=======================
============================================--------------------------------------------------------------vvvv------------------------------------------------------------------------------------------
05
# api.py

import asyncio
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from bs4 import BeautifulSoup
import requests
from sqlalchemy import text
from db import async_session, NewsItem, Journal
from floki_agent import news_agent, Runner, get_user_info, get_user_history, get_user_relevant_news

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
# Request models
# -----------------------------
class QueryRequest(BaseModel):
    user_id: str
    query: str
    n_results: int = 3

class ScrapeRequest(BaseModel):
    url: str

class SaveJournalRequest(BaseModel):
    user_id: int
    text: str
    sentiment_score: float | None = None

# -----------------------------
# ChromaDB + Sentence Transformer setup
# -----------------------------
client = chromadb.Client()
collection = client.get_or_create_collection("all_data_embeddings")
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Helper: run async in background safely
# -----------------------------
def run_async_task(coro, *args, **kwargs):
    asyncio.create_task(coro(*args, **kwargs))

# -----------------------------
# Vectorstore operations
# -----------------------------
async def fetch_and_embed_all():
    async with async_session() as session:
        texts = []

        async def batch_fetch(model_class):
            result = await session.execute(text(f"SELECT * FROM {model_class.__tablename__}"))
            rows = result.fetchall()
            for row in rows:
                text_content = ""
                if "title" in row.keys():
                    text_content += f"Title: {row.title} "
                if "summary" in row.keys():
                    text_content += f"Summary: {row.summary} "
                if "text" in row.keys():
                    text_content += f"Text: {row.text} "
                texts.append({
                    "id": f"{model_class.__name__.lower()}_{row.id}",
                    "text": text_content.strip(),
                    "title": getattr(row, "title", ""),
                    "metadata": {"id": row.id, "title": getattr(row, "title", "")}
                })

        for m in [Journal, NewsItem]:
            await batch_fetch(m)

        for doc in texts:
            vector = model.encode(doc["text"]).tolist()
            collection.add(
                documents=[doc["text"]],
                embeddings=[vector],
                ids=[doc["id"]],
                metadatas=[doc.get("metadata", {})]
            )
    logger.info(f"✅ Embedded {len(texts)} DB records successfully!")

async def scrape_and_embed_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    chunk_size = 500
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[f"website_{i}"],
            metadatas=[{"source": url}]
        )
    logger.info(f"✅ Embedded {len(chunks)} chunks from {url}")

def search_vectorstore(query: str, n_results: int = 3):
    query_vector = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

def add_documents_to_vectorstore(docs):
    for i, doc in enumerate(docs):
        vector = model.encode(doc["text"]).tolist()
        metadata = doc.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        collection.add(
            documents=[doc["text"]],
            embeddings=[vector],
            ids=[f"doc_{i}"],
            metadatas=[metadata]
        )
    logger.info(f"✅ Added {len(docs)} documents to vectorstore.")

# -----------------------------
# FastAPI endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Floki Live RAG Agent Running!"}

@app.post("/ask")
async def ask_floki(request: QueryRequest):
    try:
        try:
            vector_results = search_vectorstore(request.query, n_results=request.n_results)
            relevant_texts = " | ".join(vector_results['documents'][0]) if vector_results['documents'] else "No relevant news."
        except Exception:
            relevant_texts = "No relevant news."

        async with async_session() as session:
            try:
                result = await session.execute("SELECT title FROM news_items ORDER BY id ASC")
                rows = result.fetchall()
                history = " | ".join([row[0] for row in rows]) if rows else "No history yet."
            except Exception:
                history = "No history yet."

        user_info = await get_user_info(request.user_id)
        user_history = await get_user_history(request.user_id)
        user_news = await get_user_relevant_news(request.user_id)
        full_prompt = (
            f"{user_info}\nHistory: {user_history}\nRelevant news: {user_news}\n"
            f"Vectorstore results: {relevant_texts}\nQuery: {request.query}"
        )

        try:
            summarized_text = await Runner.run(news_agent, full_prompt)
            final_output = summarized_text.final_output
        except Exception:
            final_output = "Sorry, I couldn't fetch information. Try again later."

        return {"response": final_output}

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process request.")

@app.post("/save_journal")
async def save_journal(request: SaveJournalRequest):
    try:
        async with async_session() as session:
            new_entry = Journal(
                user_id=request.user_id,
                text=request.text,
                sentiment_score=request.sentiment_score
            )
            session.add(new_entry)
            await session.commit()
        return {"status": "✅ Journal saved!"}
    except Exception as e:
        logger.error(f"Error saving journal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save journal.")

@app.post("/update_embeddings")
async def update_embeddings(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_async_task, fetch_and_embed_all)
    return {"message": "✅ Embedding update started in background."}

@app.post("/scrape_website")
async def scrape_website(request: ScrapeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_async_task, scrape_and_embed_website, request.url)
    return {"message": f"✅ Scraping started for {request.url}"}

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki Live RAG Agent starting up...")
    try:
        await scrape_and_embed_website("https://fundedflow.app")
    except Exception as e:
        logger.error(f"Initial scrape error: {e}")

    async def periodic_update():
        while True:
            try:
                logger.info("Running periodic embedding update...")
                await fetch_and_embed_all()
            except Exception as e:
                logger.error(f"Periodic update error: {e}")
            await asyncio.sleep(600)  # 10 minutes

    asyncio.create_task(periodic_update())

# -----------------------------
# Allow independent run of vectorstore
# -----------------------------
if __name__ == "__main__":
    import sys
    if "api" in sys.argv:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    else:
        asyncio.run(fetch_and_embed_all())
        asyncio.run(scrape_and_embed_website("https://example.com"))
==================================================================================================================================


uvicorn api:app --reload

==
llmfor the embeddig 


pip install requests beautifulsoup4
pip install sentence-transformers
pip install chromadb
