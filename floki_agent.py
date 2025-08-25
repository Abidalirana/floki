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
