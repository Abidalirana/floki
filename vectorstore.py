# vectorstore.py

import asyncio
from chromadb import Client
from chromadb.utils import embedding_functions
from config import VECTORSTORE_DIR
from db import async_session, NewsItem, Journal, Emotion, Trade, FeatureUsage, ResetChallenge, RecoveryPlan, RulebookVote, SimulatorLog

# -----------------------------
# Chroma client setup (new style)
# -----------------------------
client = Client()  # No arguments; defaults to local DuckDB+Parquet

# -----------------------------
# Embedding function setup
# -----------------------------
embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_OPENAI_API_KEY_HERE",  # <-- Put your actual OpenAI key
    model_name="text-embedding-3-large"
)

# -----------------------------
# Create or get collection
# -----------------------------
collection = client.get_or_create_collection(
    name="fundedflow_embeddings",
    embedding_function=embedding_func
)

# -----------------------------
# Add documents
# -----------------------------
def add_documents_to_vectorstore(docs):
    if not docs:
        return
    collection.add(
        documents=[doc["text"] for doc in docs],
        metadatas=[{"title": doc.get("title", ""), "url": doc.get("url", "")} for doc in docs],
        ids=[str(doc["id"]) for doc in docs]
    )

# -----------------------------
# Search documents
# -----------------------------
def search_vectorstore(query, n_results=5):
    return collection.query(
        query_texts=[query],
        n_results=n_results
    )

# -----------------------------
# Fetch all DB data and embed
# -----------------------------
async def fetch_and_embed_all():
    async with async_session() as session:
        texts = []

        async def batch_fetch(model_class):
            result = await session.execute(f"SELECT * FROM {model_class.__tablename__}")
            rows = result.fetchall()
            for row in rows:
                texts.append({
                    "id": f"{model_class.__name__.lower()}_{row.id}",
                    "text": str(row.__dict__),
                    "title": getattr(row, "title", "")
                })

        models = [Journal, Emotion, Trade, FeatureUsage, ResetChallenge, RecoveryPlan, RulebookVote, SimulatorLog, NewsItem]

        for m in models:
            await batch_fetch(m)

        add_documents_to_vectorstore(texts)
        print(f"✅ Embedded {len(texts)} records successfully!")

# -----------------------------
# Run embedding if executed directly
# -----------------------------
if __name__ == "__main__":
    asyncio.run(fetch_and_embed_all())
