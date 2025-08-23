import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from floki_agent import news_agent, Runner, add_documents_to_vectorstore,  search_vectorstore
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
    n_results: Optional[int] = 3

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
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki RAG Agent starting up...")
    # Optionally run background updates
    # asyncio.create_task(periodic_news_update())

# -----------------------------
# Optional: periodic update
# -----------------------------
# async def periodic_news_update():
#     while True:
#         try:
#             await update_news()
#         except Exception as e:
#             logger.error(f"Periodic update failed: {str(e)}")
#         await asyncio.sleep(3600)  # every hour
