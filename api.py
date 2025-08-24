import asyncio
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from db import async_session, Journal
from floki_agent import (  
    search_vectorstore,
    get_user_info, 
    get_user_history, 
    get_user_relevant_news,
    fetch_and_embed_all           
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

class SaveJournalRequest(BaseModel):
    user_id: int
    text: str
    sentiment_score: float | None = None

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Floki Live RAG Agent Running!"}

@app.post("/ask")
async def ask_floki(request: QueryRequest):
    try:
        results = search_vectorstore(request.query, n_results=request.n_results)
        relevant_texts = " | ".join(results['documents'][0]) if results['documents'] else "No relevant news."

        user_info = await get_user_info(request.user_id)
        user_history = await get_user_history(request.user_id)
        user_news = await get_user_relevant_news(request.user_id)

        full_prompt = (
            f"{user_info}\nHistory: {user_history}\nRelevant news: {user_news}\n"
            f"Vectorstore results: {relevant_texts}\nQuery: {request.query}"
        )

        return {"response": full_prompt}

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
    background_tasks.add_task(fetch_and_embed_all)   # 👈 schedules website re-embedding
    return {"message": "✅ Embedding update started in background."}

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Floki Live RAG Agent starting up...")
    asyncio.create_task(fetch_and_embed_all())       # 👈 runs embeddings on boot

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
