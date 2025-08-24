from sentence_transformers import SentenceTransformer
import chromadb
from bs4 import BeautifulSoup
import requests
import asyncio
from db import async_session, NewsItem, Journal

client = chromadb.Client()
collection = client.get_or_create_collection("all_data_embeddings")
model = SentenceTransformer('all-MiniLM-L6-v2')

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

        models = [Journal, NewsItem]
        for m in models:
            await batch_fetch(m)

        for doc in texts:
            vector = model.encode(doc["text"]).tolist()
            collection.add(
                documents=[doc["text"]],
                embeddings=[vector],
                ids=[doc["id"]]
            )
    print(f"✅ Embedded {len(texts)} DB records successfully!")

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
            ids=[f"website_{i}"]
        )
    print(f"✅ Embedded {len(chunks)} chunks from {url}")

def search_vectorstore(query: str, n_results: int = 3):
    query_vector = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]  # <-- remove 'ids', add 'metadatas' if needed
    )
    return results

def add_documents_to_vectorstore(docs):
    """
    Add a list of documents to ChromaDB vectorstore.
    Each doc should be like: {"text": "...", "metadata": {"id": ..., "title": ...}}
    """
    for i, doc in enumerate(docs):
        vector = model.encode(doc["text"]).tolist()
        metadata = doc.get("metadata")
        if not metadata or not isinstance(metadata, dict):
            raise ValueError("Each document must have a non-empty metadata dictionary!")
        
        collection.add(
            documents=[doc["text"]],
            embeddings=[vector],
            metadatas=[metadata],   # <-- use metadata, not metadatas
            ids=[f"doc_{i}"]
        )
    print(f"✅ Added {len(docs)} documents to vectorstore.")


# -----------------------------
# Run embedding manually if needed
# -----------------------------
if __name__ == "__main__":
    asyncio.run(fetch_and_embed_all())
    asyncio.run(scrape_and_embed_website("https://example.com"))
