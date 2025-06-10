import os
from dotenv import load_dotenv
load_dotenv()

import base64
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from qa_pipeline import load_vectorstore, embed_text, answer_question

app = FastAPI()

# Load the FAISS vectorstore once on startup
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
