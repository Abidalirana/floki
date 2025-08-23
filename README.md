We’ll use ChromaDB (local vector DB) for simplicity, openai embeddings, and your FastAPI agent.

Folder Structure (Updated for RAG)
floki_agent/
│
├─ .venv/
├─ .env
├─ .gitignore
├─ .python-version
├─ api.py
├─ config.py
├─ db.py
├─ floki_agent.py      #is a full AI assistant agent:
├─ vectorstore.py      # New: RAG vector DB + embedding+ loaiwe all inside of it
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
4️⃣ ingest.py

We’ll fetch a few sample finance headlines:

import aiohttp
from bs4 import BeautifulSoup
from typing import List
from db import NewsItem, async_session

NEWS_SOURCES = [
    "https://www.cnbc.com/finance/",
    "https://www.reuters.com/finance/"
]

async def fetch_html(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

async def scrape_articles() -> List[NewsItem]:
    articles = []
    for source in NEWS_SOURCES:
        html = await fetch_html(source)
        soup = BeautifulSoup(html, "html.parser")
        headlines = soup.select("a")[:5]
        for link in headlines:
            title = link.get_text(strip=True)
            url = link.get("href")
            if title and url:
                articles.append(NewsItem(title=title, url=url))
    async with async_session() as session:
        session.add_all(articles)
        await session.commit()
    return articles

async def run_collector() -> List[NewsItem]:
    return await scrape_articles()
==============================---------------------------------------------------------------------

==============================---------------------------------------------------------------------

05 vectorstore.py
# vectorstore.py

import asyncio
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import VECTORSTORE_DIR
from db import async_session, NewsItem, Journal, Emotion, Trade, FeatureUsage, ResetChallenge, RecoveryPlan, RulebookVote, SimulatorLog

# -----------------------------
# Chroma client setup
# -----------------------------
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=VECTORSTORE_DIR
))

embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=None,  # Add your Gemini/OpenAI API key here
    model_name="text-embedding-3-large"
)

collection = client.get_or_create_collection(
    name="fundedflow_embeddings",
    embedding_function=embedding_func
)

# -----------------------------
# Add documents
# -----------------------------
def add_documents_to_vectorstore(docs):
    for doc in docs:
        collection.add(
            documents=[doc["text"]],
            metadatas=[{"title": doc.get("title", ""), "url": doc.get("url", "")}],
            ids=[str(doc["id"])]
        )

# -----------------------------
# Search documents
# -----------------------------
def search_vectorstore(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

# -----------------------------
# Fetch all data from DB and embed
# -----------------------------
async def fetch_and_embed_all():
    async with async_session() as session:
        texts = []

        async def batch_fetch(model_class):
            result = await session.execute(f"SELECT * FROM {model_class.__tablename__}")
            for row in result.fetchall():
                texts.append({
                    "id": f"{model_class.__name__.lower()}_{row.id}",
                    "text": str(row.__dict__),
                    "title": getattr(row, "title", "")
                })

        # List all models you want to embed
        models = [Journal, Emotion, Trade, FeatureUsage, ResetChallenge, RecoveryPlan, RulebookVote, SimulatorLog, NewsItem]

        for m in models:
            await batch_fetch(m)

        # Add all to vectorstore
        add_documents_to_vectorstore(texts)
        print(f"✅ Embedded {len(texts)} records successfully!")

# -----------------------------
# Run embedding if called directly
# -----------------------------
if __name__ == "__main__":
    asyncio.run(fetch_and_embed_all())

==============================---------------------------------------------------------------------
06 floki_agent_gemini.py

import asyncio
from ingest import run_collector
from vectorstore import add_documents_to_vectorstore, search_vectorstore
from db import NewsItem, async_session
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
# Use Floki system instruction exactly as provided
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
# Model setup with Floki instructions
# -----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
    system_instruction=FLOKI_INSTRUCTION
)

# -----------------------------
# Tools
# -----------------------------
@function_tool
async def summarize_text_tool(text: str) -> str:
    prompt = f"Summarize this text in 3 bullet points:\n{text}"
    resp = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

@function_tool
async def get_user_info(user_id: str) -> str:
    return f"User {user_id} | Name: Abid | Role: Reader"

@function_tool
async def get_user_history(user_id: str) -> str:
    async with async_session() as session:
        result = await session.execute(
            "SELECT title FROM news_items ORDER BY id DESC LIMIT 5"
        )
        rows = result.fetchall()
        return " | ".join([row[0] for row in rows]) if rows else "No history yet."

@function_tool
async def get_user_relevant_news(user_id: str) -> str:
    results = search_vectorstore("stocks", n_results=3)
    return " | ".join(results['documents'][0]) if results['documents'] else "No news found."

# -----------------------------
# Floki Agent
# -----------------------------
news_agent = Agent(
    name="FlokiNewsAssistant",
    instructions=FLOKI_INSTRUCTION,
    tools=[
        summarize_text_tool,
        get_user_info,
        get_user_history,
        get_user_relevant_news
    ],
    model=model
)

# -----------------------------
# Main async workflow
# -----------------------------
async def main():
    user_id = "1234"  # example

    # Step 1: Collect news
    articles = await run_collector()
    print(f"Collected {len(articles)} articles")

    # Step 2: Summarize using Floki agent
    news_texts = [f"{idx+1}. {n.title}" for idx, n in enumerate(articles)]
    full_text = "\n".join(news_texts)

    user_info = await get_user_info(user_id)
    history = await get_user_history(user_id)
    relevant_news = await get_user_relevant_news(user_id)

    full_prompt = f"{user_info}\nHistory: {history}\nRelevant news: {relevant_news}\nNews to summarize:\n{full_text}"

    summarized_text = await Runner.run(news_agent, full_prompt)
    print("Summary output from Floki Gemini:\n", summarized_text.final_output)

    # Step 3: Add to vectorstore
    docs = [{"id": art.id, "title": art.title, "text": art.title} for art in articles]
    add_documents_to_vectorstore(docs)
    print("Added to vectorstore.")

    # Step 4: Example vector search
    query_results = search_vectorstore("stocks", n_results=3)
    if query_results['documents']:
        print("Vectorstore search result:", query_results['documents'][0])

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())



============================---------------------------------------------------------------------============================---------------------------------------------------------------------
07
# vectorstore.py

import asyncio
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import VECTORSTORE_DIR
from db import async_session, NewsItem, Journal, Emotion, Trade, FeatureUsage, ResetChallenge, RecoveryPlan, RulebookVote, SimulatorLog

# -----------------------------
# Chroma client setup
# -----------------------------
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=VECTORSTORE_DIR
))

embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=None,  # Add your Gemini/OpenAI API key here
    model_name="text-embedding-3-large"
)

collection = client.get_or_create_collection(
    name="fundedflow_embeddings",
    embedding_function=embedding_func
)

# -----------------------------
# Add documents
# -----------------------------
def add_documents_to_vectorstore(docs):
    for doc in docs:
        collection.add(
            documents=[doc["text"]],
            metadatas=[{"title": doc.get("title", ""), "url": doc.get("url", "")}],
            ids=[str(doc["id"])]
        )

# -----------------------------
# Search documents
# -----------------------------
def search_vectorstore(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

# -----------------------------
# Fetch all data from DB and embed
# -----------------------------
async def fetch_and_embed_all():
    async with async_session() as session:
        texts = []

        async def batch_fetch(model_class):
            result = await session.execute(f"SELECT * FROM {model_class.__tablename__}")
            for row in result.fetchall():
                texts.append({
                    "id": f"{model_class.__name__.lower()}_{row.id}",
                    "text": str(row.__dict__),
                    "title": getattr(row, "title", "")
                })

        # List all models you want to embed
        models = [Journal, Emotion, Trade, FeatureUsage, ResetChallenge, RecoveryPlan, RulebookVote, SimulatorLog, NewsItem]

        for m in models:
            await batch_fetch(m)

        # Add all to vectorstore
        add_documents_to_vectorstore(texts)
        print(f"✅ Embedded {len(texts)} records successfully!")

# -----------------------------
# Run embedding if called directly
# -----------------------------
if __name__ == "__main__":
    asyncio.run(fetch_and_embed_all())









============================================--------------------------------------------------------------vvvv------------------------------------------------------------------------------------------

08
api.py 

import asyncio
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from floki_agent_gemini import news_agent, Runner, add_documents_to_vectorstore, run_collector, search_vectorstore
from db import async_session, NewsItem
import logging

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Floki Gemini RAG Agent")

# -----------------------------
# Request body schema
# -----------------------------
class QueryRequest(BaseModel):
    user_id: str
    query: str
    n_results: Optional[int] = 3  # Optional: number of relevant news items

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Floki Gemini RAG Agent Running!"}

# -----------------------------
# Ask Floki endpoint
# -----------------------------
@app.post("/ask")
async def ask_floki(request: QueryRequest):
    try:
        # Step 1: Vectorstore search
        vector_results = search_vectorstore(request.query, n_results=request.n_results)
        relevant_texts = " | ".join(vector_results['documents'][0]) if vector_results['documents'] else "No relevant news found."

        # Step 2: Fetch last 5 news from DB as history
        async with async_session() as session:
            result = await session.execute("SELECT title FROM news_items ORDER BY id DESC LIMIT 5")
            rows = result.fetchall()
            history = " | ".join([row[0] for row in rows]) if rows else "No history yet."

        # Step 3: Combine prompt
        prompt = f"User {request.user_id} Query: {request.query}\nHistory: {history}\nRelevant news: {relevant_texts}"

        # Step 4: Run Floki agent
        response = await Runner.run(news_agent, prompt)
        return {"response": response.final_output}

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process your request.")

# -----------------------------
# Update news DB + vectorstore
# -----------------------------
@app.post("/update_news")
async def update_news():
    try:
        articles = await run_collector()
        if not articles:
            return {"message": "No new articles collected."}

        # Add to vectorstore
        docs = [{"id": art.id, "title": art.title, "text": art.title} for art in articles]
        add_documents_to_vectorstore(docs)
        logger.info(f"Collected {len(articles)} articles and added to vectorstore.")
        return {"message": f"Collected {len(articles)} articles and added to vectorstore."}

    except Exception as e:
        logger.error(f"Error in /update_news: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update news.")

# -----------------------------
# Background task example (optional)
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki RAG Agent starting up...")
    # Optionally run background news update every X seconds/minutes
    # asyncio.create_task(periodic_news_update())

# async def periodic_news_update():
#     while True:
#         try:
#             await update_news()
#         except Exception as e:
#             logger.error(f"Periodic update failed: {str(e)}")
#         await asyncio.sleep(3600)  # every hour
==========





uvicorn api:app --reload