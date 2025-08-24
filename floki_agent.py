import asyncio
from datetime import datetime, timezone
import chromadb
from sentence_transformers import SentenceTransformer
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from openai import AsyncOpenAI
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Disable tracing for cleaner logs
# -----------------------------
set_tracing_disabled(True)

# -----------------------------
# Setup ChromaDB & Embeddings
# -----------------------------
client = chromadb.Client()
collection = client.get_or_create_collection("floki_embeddings")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Floki System Instruction
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
# OpenAI Gemini Client
# -----------------------------
client_ai = AsyncOpenAI(
    api_key="YOUR_GEMINI_API_KEY",   # 🔑 replace with your key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -----------------------------
# Model setup
# -----------------------------
model_ai = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client_ai,
)

# -----------------------------
# Vectorstore Helpers
# -----------------------------
def add_to_vectorstore(text: str, metadata: dict = None):
    vector = model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[vector],
        ids=[str(hash(text))],
        metadatas=[metadata or {}]
    )

def search_vectorstore(query: str, n_results: int = 3):
    query_vector = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

# -----------------------------
# Agent Tools
# -----------------------------
async def summarize_text(text: str) -> str:
    resp = await client_ai.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": f"Summarize this text in 3 bullet points:\n{text}"}],
        temperature=0.3
    )
    return resp.choices[0].message.content

async def get_user_info(user_id: str) -> str:
    return f"User {user_id} | Name: Abid | Role: Reader"

async def get_user_history(user_id: str) -> str:
    results = search_vectorstore("trading", n_results=3)
    return " | ".join(results['documents'][0]) if results['documents'] else "No history yet."

async def get_user_relevant_news(user_id: str) -> str:
    results = search_vectorstore("stocks", n_results=3)
    return " | ".join(results['documents'][0]) if results['documents'] else "No news found."

# Register as tools
summarize_text_tool = function_tool(summarize_text)
get_user_info_tool = function_tool(get_user_info)
get_user_history_tool = function_tool(get_user_history)
get_user_relevant_news_tool = function_tool(get_user_relevant_news)

# -----------------------------
# Floki Agent
# -----------------------------
floki_agent = Agent(
    name="FlokiNewsAssistant",
    instructions=FLOKI_INSTRUCTION,
    tools=[
        summarize_text_tool,
        get_user_info_tool,
        get_user_history_tool,
        get_user_relevant_news_tool,
    ],
    model=model_ai
)

# -----------------------------
# Interactive CLI loop
# -----------------------------
async def interactive_loop():
    user_id = "1234"

    print("🟢 Floki Agent Interactive Testing Started! Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() in ["exit", "quit"]:
            print("🛑 Exiting Floki Agent testing loop.")
            break

        # Run agent
        response = await Runner.run(floki_agent, query)
        print("\n💬 Floki Agent Response:\n", response.final_output)

        # ✅ Save query in vectorstore
        add_to_vectorstore(query, {
            "user_id": user_id,
            "source": "interactive_loop",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        print("✅ Query added to vectorstore.\n")

# -----------------------------
# NEW: Website Fetch & Embed
# -----------------------------
async def fetch_and_embed_all():
    url = "https://fundedflow.app"   # 👈 changed here
    print(f"🌐 Fetching content from {url} ...")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        texts = [t for t in paragraphs if len(t) > 30]  # filter small junk

        if not texts:
            print("⚠️ No text extracted.")
            return

        for text in texts:
            add_to_vectorstore(text, {"source": url, "timestamp": datetime.now(timezone.utc).isoformat()})

        print(f"✅ Embedded {len(texts)} paragraphs from {url}")

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    asyncio.run(interactive_loop())
