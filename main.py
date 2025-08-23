async def main():
    user_id = "1234"

    # Step 1: Collect news
    articles = await run_collector()
    print(f"Collected {len(articles)} articles")

    # Step 2: Fetch user context
    user_info = await get_user_info(user_id)
    history = await get_user_history(user_id)
    relevant_news = await get_user_relevant_news(user_id)

    # Step 3: Combine news with user context
    news_texts = [f"{idx+1}. {n.title}" for idx, n in enumerate(articles)]
    full_news_text = "\n".join(news_texts)
    agent_input = f"{user_info}\nHistory: {history}\nRelevant news: {relevant_news}\nNews to summarize:\n{full_news_text}"

    # Step 4: Run agent
    summarized_text = await Runner.run(news_agent, agent_input)
    print("Summary output from Gemini:\n", summarized_text.final_output)

    # Step 5: Add to vectorstore
    docs = [{"id": art.id, "title": art.title, "text": art.title} for art in articles]
    add_documents_to_vectorstore(docs)
    print("Added to vectorstore.")

    # Step 6: Example vector query
    query_results = search_vectorstore("stocks", n_results=3)
    print("Vectorstore search result:", query_results.get('documents', ["No results"])[0])
