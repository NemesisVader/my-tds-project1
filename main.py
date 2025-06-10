import os
from dotenv import load_dotenv
load_dotenv()

import base64
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

from qa_pipeline import load_vectorstore, embed_text, answer_question

app = FastAPI()

# --- CORS Configuration ---
# Define the origins that are allowed to access your API
# You should replace "*" with the actual domain(s) of your frontend application
# For development, "*" is fine, but for production, specify explicit domains.
origins = [
    "https://exam.sanand.workers.dev",  # The evaluation website
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)
# --- End CORS Configuration ---


# Load the FAISS vectorstore once on startup
# Make sure your FAISS index and embedding model are accessible in the Vercel environment.
# This might involve placing them in the same directory as your API or configuring Vercel to fetch them.
vectorstore = load_vectorstore()

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded string

class LinkItem(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[LinkItem]

@app.post("/", response_model=QuestionResponse)
async def ask_question(data: QuestionRequest):
    try:
        answer, sources = answer_question(data.question, vectorstore)

        # sources is a list of dicts with keys 'url' and 'text'
        links = list(sources)

        # ✅ Ensure JSON content type with explicit JSONResponse
        return JSONResponse(
            content={
                "answer": answer,
                "links": links
            },
            media_type="application/json"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )