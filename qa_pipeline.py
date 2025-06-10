# === Updated qa_pipeline.py ===
import json
from dotenv import load_dotenv
load_dotenv()

import os
import httpx
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# === AI Proxy credentials ===
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai"
print("🔐 AIPROXY_TOKEN from os.environ:", os.environ.get("AIPROXY_TOKEN"))

# === Manual embedding function ===
def embed_text(texts):
    url = f"{AIPROXY_BASE_URL}/v1/embeddings"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

# === Load FAISS vectorstore ===
def load_vectorstore(path="vectorstore"):
    dummy_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="sk-dummy-key",
        base_url="https://aiproxy.sanand.workers.dev/openai/v1"
    )
    return FAISS.load_local(path, dummy_embeddings, allow_dangerous_deserialization=True)

# === Ask question using vector search + GPT ===
def answer_question(question, vectorstore, k=5):
    question_vector = embed_text([question])[0]
    docs = vectorstore.similarity_search_by_vector(question_vector, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    url = f"{AIPROXY_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful Virtual TA for the Tools for Data Science (TDS) course. "
                    "Answer clearly based on the following context:\n\n"
                    f"{context}\n\n"
                    "If the context contains source links, return a JSON like:\n"
                    "{\n  \"answer\": \"...\",\n  \"links\": [\n    {\"url\": \"...\", \"text\": \"...\"}, ...\n  ]\n}\n"
                    "Otherwise, just return a plain text answer."
                )
            },
            {"role": "user", "content": question}
        ]
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        raw_answer = result["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(raw_answer)
        answer = parsed["answer"]
        links = parsed.get("links", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        # fallback to just text + links from FAISS docs
        answer = raw_answer
        links = []
        for doc in docs:
            url = doc.metadata.get("source", "Unknown")
            snippet = doc.page_content.strip().split("\n")[0][:300]
            links.append({
                "url": url,
                "text": snippet
            })

    return answer, links

