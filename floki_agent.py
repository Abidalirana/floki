#floki_agent.py

import asyncio

from vectorstore import add_documents_to_vectorstore, search_vectorstore
from db import NewsItem, async_session
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
# Use Floki system instruction exactly as provided
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
# Model setup with Floki instructions
# -----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
    
)

# -----------------------------
# Tools
# -----------------------------
@function_tool
async def summarize_text_tool(text: str) -> str:
    prompt = f"Summarize this text in 3 bullet points:\n{text}"
    resp = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

@function_tool
async def get_user_info(user_id: str) -> str:
    return f"User {user_id} | Name: Abid | Role: Reader"

@function_tool
async def get_user_history(user_id: str) -> str:
    async with async_session() as session:
        result = await session.execute(
            "SELECT title FROM news_items ORDER BY id DESC LIMIT 5"
        )
        rows = result.fetchall()
        return " | ".join([row[0] for row in rows]) if rows else "No history yet."

@function_tool
async def get_user_relevant_news(user_id: str) -> str:
    results = search_vectorstore("stocks", n_results=3)
    return " | ".join(results['documents'][0]) if results['documents'] else "No news found."

# -----------------------------
# Floki Agent
# -----------------------------
news_agent = Agent(
    name="FlokiNewsAssistant",
    instructions=FLOKI_INSTRUCTION,
    tools=[
        summarize_text_tool,
        get_user_info,
        get_user_history,
        get_user_relevant_news
    ],
    model=model
)

# -----------------------------
# Main async workflow
# -----------------------------
async def main():
    user_id = "1234"  # example

    # Step 1: Collect news
    articles = await run_collector()
    print(f"Collected {len(articles)} articles")

    # Step 2: Summarize using Floki agent
    news_texts = [f"{idx+1}. {n.title}" for idx, n in enumerate(articles)]
    full_text = "\n".join(news_texts)

    user_info = await get_user_info(user_id)
    history = await get_user_history(user_id)
    relevant_news = await get_user_relevant_news(user_id)

    full_prompt = f"{user_info}\nHistory: {history}\nRelevant news: {relevant_news}\nNews to summarize:\n{full_text}"

    summarized_text = await Runner.run(news_agent, full_prompt)
    print("Summary output from Floki Gemini:\n", summarized_text.final_output)

    # Step 3: Add to vectorstore
    docs = [{"id": art.id, "title": art.title, "text": art.title} for art in articles]
    add_documents_to_vectorstore(docs)
    print("Added to vectorstore.")

    # Step 4: Example vector search
    query_results = search_vectorstore("stocks", n_results=3)
    if query_results['documents']:
        print("Vectorstore search result:", query_results['documents'][0])

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
