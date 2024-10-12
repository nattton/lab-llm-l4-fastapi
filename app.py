import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai import AsyncOpenAI


load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###########
# Models #
########################################################################################


class GrammarTaskRequest(BaseModel):
    text: str
    style: Optional[str] = "default"


class GrammarTaskResponse(BaseModel):
    text: str


###########
# Routers #
########################################################################################


@app.get("/")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/openai/grammar")
async def openai_grammar(request: GrammarTaskRequest) -> GrammarTaskResponse:
    style_prompts = {
        "informal": "Using informal words like talking to a friend.",
        "ielts": "Using very fancy words.",
        "formal": "Using formal words.",
        "acedemic": "Using academic words suitable for acedemic publications.",
        "default": ""
    }

    text = request.text
    style = request.style
    response = await client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You will be provided with statements, and your task is to convert them to standard English." + style_prompts[style]
        },
        {
            "role": "user",
            "content": text
        }
    ],
        temperature=0.7,
        max_tokens=64,
        top_p=1
    )

    return {"text": response.choices[0].message.content}

########################################################################################
