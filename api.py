import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from vectorstore import fetch_and_embed_all, scrape_and_embed_website, search_vectorstore
from db import async_session, NewsItem, Journal
import logging

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
# Ask Floki endpoint
# -----------------------------
@app.post("/ask")
async def ask_floki(request: QueryRequest):
    try:
        # Step 1: Search vectorstore
        vector_results = search_vectorstore(request.query, n_results=request.n_results)
        relevant_texts = " | ".join(vector_results['documents'][0]) if vector_results['documents'] else "No relevant news."

        # Step 2: Get last 5 news as history
        async with async_session() as session:
            result = await session.execute("SELECT title FROM news_items ORDER BY id DESC LIMIT 5")
            rows = result.all()
            history = " | ".join([row[0] for row in rows]) if rows else "No history yet."

        response = f"User {request.user_id} Query: {request.query}\nHistory: {history}\nRelevant news: {relevant_texts}"
        return {"response": response}

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
            new_entry = Journal(user_id=request.user_id, text=request.text, sentiment_score=request.sentiment_score)
            session.add(new_entry)
            await session.commit()
        return {"status": "✅ Journal saved!"}
    except Exception as e:
        logger.error(f"Error saving journal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save journal.")

# -----------------------------
# Background tasks: update embeddings
# -----------------------------
@app.post("/update_embeddings")
async def update_embeddings(background_tasks: BackgroundTasks):
    background_tasks.add_task(fetch_and_embed_all)
    return {"message": "✅ Embedding update started in background."}

# -----------------------------
# Background tasks: scrape website
# -----------------------------
@app.post("/scrape_website")
async def scrape_website(request: ScrapeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(scrape_and_embed_website, request.url)
    return {"message": f"✅ Scraping started for {request.url}"}

# -----------------------------
# Startup event: optionally run periodic tasks
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki Live RAG Agent starting up...")

    # Scrape your website once at startup
    website_url = "https://your-website.com"  # <-- put your full website here
    asyncio.create_task(scrape_and_embed_website(website_url))

    # Periodic updates
    async def periodic_update():
        while True:
            try:
                logger.info("Running periodic embedding update...")
                await fetch_and_embed_all()  # fetch & embed all data from Supabase + other sources
            except Exception as e:
                logger.error(f"Periodic update error: {str(e)}")
            await asyncio.sleep(60*10)  # every 10 minutes

    asyncio.create_task(periodic_update())

#=============================
