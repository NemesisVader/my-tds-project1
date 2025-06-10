# main.py (or your FastAPI entry file)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

# Add this entire origins section to allow requests
# from the evaluation website and local development.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define your data models using Pydantic
class Link(BaseModel):
    text: str
    url: str

class RequestBody(BaseModel):
    question: str
    image: str | None = None # Optional image field

class ResponseBody(BaseModel):
    answer: str
    links: List[Link]


@app.post("/")
def handle_request(body: RequestBody) -> ResponseBody:
    # Your API logic here
    # This example returns the static response from the prompt
    return ResponseBody(
      answer="You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
      links=[
        {
          "text": "Use the model that’s mentioned in the question.",
          "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4"
        },
        {
          "text": "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate.",
          "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3"
        }
      ]
    )