from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import tempfile
import shutil

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    topic: str
    transcript: str
    history: list
    

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return {"transcript": transcript.text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)

@app.post("/respond")
async def respond(req: ChatRequest):
    try:
        prompt = (
            f"You are an AI interviewer for the topic: {req.topic}. "
            "You must ask only one concise interview question at a time. "
            "Always respond in English. "
            "Politely steer the user back if they go off-topic."
        )
        
        # If user transcript is empty, start the interview with first question
        if req.transcript.strip() == "":
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a friendly AI interviewer for the topic: {req.topic}. "
                        "Start with a short welcome and brief introduction to the topic, but do not ask any interview questions yet. "
                        "Keep it under 40 words. Wait for the candidate to indicate they are ready before asking anything. Do not use more than 2 sentences. Do not mention that you are OpenAI."
                    )
                }
            ]
        else:
            messages = [
                {"role": "system", "content": prompt},
                *req.history,
                {"role": "user", "content": req.transcript}
            ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        reply = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        return {"error": str(e)}
