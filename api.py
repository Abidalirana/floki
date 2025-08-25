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
