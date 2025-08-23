import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from db import async_session, NewsItem, Journal    # add other models if needed
from vectorstore import fetch_and_embed_all, scrape_and_embed_website, search_vectorstore 
import logging

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Floki Offline RAG Agent")

# -----------------------------
# Request body schema
# -----------------------------
class QueryRequest(BaseModel):
    user_id: str
    query: str
    n_results: Optional[int] = 3

class ScrapeRequest(BaseModel):
    url: str

class SaveJournalRequest(BaseModel):
    user_id: int
    text: str
    sentiment_score: Optional[float] = None

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Floki Offline RAG Agent Running!"}

# -----------------------------
# Ask Floki endpoint (offline search only)
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
            rows = result.all()  # async fetch
            history = " | ".join([row[0] for row in rows]) if rows else "No history yet."

        # Step 3: Combine response (offline)
        combined_response = f"User {request.user_id} Query: {request.query}\nHistory: {history}\nRelevant news: {relevant_texts}"
        return {"response": combined_response}

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process your request.")

# -----------------------------
# Save journal endpoint
# -----------------------------
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
        return {"status": "✅ Journal entry saved successfully!"}
    except Exception as e:
        logger.error(f"Error in /save_journal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save journal entry.")

# -----------------------------
# Update DB embeddings (background task)
# -----------------------------
@app.post("/update_embeddings")
async def update_embeddings(background_tasks: BackgroundTasks):
    try:
        # Run embedding in background
        background_tasks.add_task(fetch_and_embed_all)
        return {"message": "✅ Embedding task started in background."}
    except Exception as e:
        logger.error(f"Error in /update_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start embedding task.")

# -----------------------------
# Scrape website & embed (background task)
# -----------------------------
@app.post("/scrape_website")
async def scrape_website(request: ScrapeRequest, background_tasks: BackgroundTasks):
    try:
        # Run scraping in background
        background_tasks.add_task(scrape_and_embed_website, request.url)
        return {"message": f"✅ Website scraping task started for {request.url}."}
    except Exception as e:
        logger.error(f"Error in /scrape_website: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start scraping task.")

# -----------------------------
# Optional chat root
# -----------------------------
@app.get("/chat")
async def chat_root():
    return {"message": "Use POST /ask to query the Floki agent."}

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki Offline RAG Agent starting up...")
