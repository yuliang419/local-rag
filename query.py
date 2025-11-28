import os
from langchain_ollama import ChatOllama
from werkzeug.datastructures import FileStorage
from get_vector_db import get_vector_db
from embed import embed

LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")


def retrieve_context(query: str):
    """Retrieve information from the Chroma DB."""
    db = get_vector_db()
    retrieved_docs = db.similarity_search(query, k=3)

    context = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return context


def file_from_path(path):
    # Open the file as a binary stream
    f = open(path, "rb")

    # Wrap it so it looks like an uploaded file
    return FileStorage(
        stream=f,
        filename=os.path.basename(path),
        content_type="application/pdf"
    )


if __name__ == "__main__":
    file = file_from_path("/Users/yuliang/Documents/Liang_Yu_CV_Europe.pdf")
    embed(file)
    
    model = ChatOllama(model=LLM_MODEL, format=None)

    query = "Who is Liang Yu?"

    # Retrieve context from the vector DB
    context = retrieve_context(query)

    # Build RAG prompt
    rag_prompt = f"""
You are a helpful assistant. Use the provided context to answer the user's question.
If the context does not provide enough information, say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    # Generate answer
    response = model.invoke(rag_prompt)

    print("Answer:", response.content)
