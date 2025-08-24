from vectorstore import add_documents_to_vectorstore, search_vectorstore, scrape_and_embed_website

# Add a sample document
from vectorstore import add_documents_to_vectorstore

# Make sure each document has a non-empty metadata dict
doc = [
    {
        "text": "Floki agent is learning how to use embedded data.",
        "metadata": {"source": "test_doc", "author": "abid"}  # <-- non-empty metadata
    }
]

add_documents_to_vectorstore(doc)


# Search to confirm embedding
results = search_vectorstore("embedded data")
print("Search results:", results)
