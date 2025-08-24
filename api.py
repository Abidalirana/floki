# api.py

import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from vectorstore import fetch_and_embed_all, scrape_and_embed_website, search_vectorstore, add_documents_to_vectorstore
from db import async_session, Journal
from floki_agent import news_agent, Runner, get_user_info, get_user_history, get_user_relevant_news
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Health check
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Floki Live RAG Agent Running!"}

# -----------------------------
# Ask Floki
# -----------------------------
@app.post("/ask")
async def ask_floki(request: QueryRequest):
    try:
        # Safe vectorstore search
        try:
            vector_results = search_vectorstore(request.query, n_results=request.n_results)
            relevant_texts = " | ".join(vector_results['documents'][0]) if vector_results['documents'] else "No relevant news."
        except Exception:
            relevant_texts = "No relevant news."

        # Safe DB history
        async with async_session() as session:
            try:
                result = await session.execute("SELECT title FROM news_items ORDER BY id ASC")
                rows = result.fetchall()
                history = " | ".join([row[0] for row in rows]) if rows else "No history yet."
            except Exception:
                history = "No history yet."

        # Compose prompt
        user_info = await get_user_info(request.user_id)
        user_history = await get_user_history(request.user_id)
        user_news = await get_user_relevant_news(request.user_id)
        full_prompt = (
            f"{user_info}\n"
            f"History: {user_history}\n"
            f"Relevant news: {user_news}\n"
            f"Vectorstore results: {relevant_texts}\n"
            f"Query: {request.query}"
        )

        # Safe agent call
        try:
            summarized_text = await Runner.run(news_agent, full_prompt)
            final_output = summarized_text.final_output
        except Exception:
            final_output = "Sorry, I couldn't fetch information. Try again later."

        # Save query to vectorstore safely
        try:
            add_documents_to_vectorstore([
                {
                    "text": request.query,
                    "metadatas": {
                        "user_id": request.user_id,
                        "source": "ask_endpoint",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            ])
        except Exception:
            pass

        return {"response": final_output}

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process request.")

# -----------------------------
# Save journal
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
        return {"status": "✅ Journal saved!"}
    except Exception as e:
        logger.error(f"Error saving journal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save journal.")

# -----------------------------
# Background tasks
# -----------------------------
@app.post("/update_embeddings")
async def update_embeddings(background_tasks: BackgroundTasks):
    background_tasks.add_task(fetch_and_embed_all)
    return {"message": "✅ Embedding update started in background."}

@app.post("/scrape_website")
async def scrape_website(request: ScrapeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(scrape_and_embed_website, request.url)
    return {"message": f"✅ Scraping started for {request.url}"}

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki Live RAG Agent starting up...")

    # Embed FundedFlow website at launch to ensure vectorstore has real data
   
    
    await scrape_and_embed_website("https://fundedflow.app")

    # Periodic background update every 10 minutes
    async def periodic_update():
        while True:
            try:
                logger.info("Running periodic embedding update...")
                await fetch_and_embed_all()
            except Exception as e:
                logger.error(f"Periodic update error: {str(e)}")
            await asyncio.sleep(60 * 10)

    asyncio.create_task(periodic_update())
