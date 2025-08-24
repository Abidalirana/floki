# floki_agent.py

import asyncio
from sqlalchemy import text
from datetime import datetime, timezone

from vectorstore import fetch_and_embed_all, scrape_and_embed_website, search_vectorstore, add_documents_to_vectorstore
from db import async_session
from config import GEMINI_API_KEY
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from openai import AsyncOpenAI

# -----------------------------
# Disable tracing for cleaner logs
# -----------------------------
set_tracing_disabled(True)

# -----------------------------
# Gemini client (AsyncOpenAI)
# -----------------------------
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -----------------------------
# Floki system instruction
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
# Model setup
# -----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)

# -----------------------------
# Async helper functions
# -----------------------------
async def summarize_text(text: str) -> str:
    resp = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": f"Summarize this text in 3 bullet points:\n{text}"}],
        temperature=0.3
    )
    return resp.choices[0].message.content

async def get_user_info(user_id: str) -> str:
    return f"User {user_id} | Name: Abid | Role: Reader"

async def get_user_history(user_id: str) -> str:
    async with async_session() as session:
        result = await session.execute(text("SELECT title FROM news_items ORDER BY id DESC LIMIT 5"))
        rows = result.fetchall()
        return " | ".join([row[0] for row in rows]) if rows else "No history yet."

async def get_user_relevant_news(user_id: str) -> str:
    results = search_vectorstore("stocks", n_results=3)
    return " | ".join(results['documents'][0]) if results['documents'] else "No news found."

# -----------------------------
# Register functions as Agent tools
# -----------------------------
summarize_text_tool = function_tool(summarize_text)
get_user_info_tool = function_tool(get_user_info)
get_user_history_tool = function_tool(get_user_history)
get_user_relevant_news_tool = function_tool(get_user_relevant_news)

# -----------------------------
# Floki Agent
# -----------------------------
news_agent = Agent(
    name="FlokiNewsAssistant",
    instructions=FLOKI_INSTRUCTION,
    tools=[
        summarize_text_tool,
        get_user_info_tool,
        get_user_history_tool,
        get_user_relevant_news_tool,
    ],
    model=model
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

        # Fetch info/history/news
        user_info = await get_user_info(user_id)
        history = await get_user_history(user_id)
        relevant_news = await get_user_relevant_news(user_id)
        full_prompt = f"{user_info}\nHistory: {history}\nRelevant news: {relevant_news}\nQuery: {query}"

        # Run agent
        summarized_text = await Runner.run(news_agent, full_prompt)
        print("\n💬 Floki Agent Response:\n", summarized_text.final_output)

        # ✅ Add last query to vectorstore (fixed metadata)
        add_documents_to_vectorstore([
            {
                "text": query,
                "metadata": {  # ✅ singular key
                    "user_id": user_id,
                    "source": "interactive_loop",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        ])
        print("✅ Query added to vectorstore.\n")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    asyncio.run(interactive_loop())
