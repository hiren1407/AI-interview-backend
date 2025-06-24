from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import tempfile
import shutil
import fitz 

from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import io
load_dotenv()
import base64

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # for local dev
        "https://tech-ai-interview.vercel.app"  # for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    topic: str
    transcript: str
    history: list
    session_id: str

resume_store = {}
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...),session_id: str = Form(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Extract text using PyMuPDF
        with fitz.open(tmp_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()

        os.remove(tmp_path)

        resume_store[session_id] = text
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), session_id: str = Form(...)):
    unique_name = f"{session_id}_{file.filename}"
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
        if req.topic == "Resume Based Questions":
            resume_text = resume_store.get(req.session_id)
            prompt = (
                "You are an AI interviewer asking resume-based questions. "
                "Always respond in English. Use the following resume:\n\n"
                f"{resume_text}\n\n"
                "Start by asking questions based on the candidate's resume. "
                "After each response, offer brief feedback(1 sentence) and ask the next question. "
                "Avoid ending the interview early."
            )
        else:
            prompt = (
                f"You are a friendly and professional AI interviewer for the topic: {req.topic}. "
                "Always respond in English. "
                "After each candidate response, first provide a brief encouraging or constructive comment (1 sentence), "
                "then immediately ask the next concise and relevant interview question. "
                "Avoid summarizing the entire interview or ending the session unless explicitly asked. "
                "Do not say 'Do you have any questions?' or 'Thank you for your time' unless the user clearly signals the interview is over. "
                "If the user goes off-topic, gently steer them back with a polite reminder."
            )
        
        if req.transcript.strip() == "":
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Welcome! Let's begin your interview on {req.topic}. "
                        "Let me know when you're ready to start."
                    )
                }
            ]
            reply_text = messages[0]["content"]
        else:
            messages = [
                {"role": "system", "content": prompt},
                *req.history,
                {"role": "user", "content": f"The candidate answered: \"{req.transcript}\""}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4
            )

            reply_text = response.choices[0].message.content

        # Generate TTS audio for the reply_text
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text
        )
        audio_bytes = tts_response.content

        # Convert audio bytes to base64 string for JSON transport
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {
            "reply": reply_text,
            "audio_base64": audio_base64
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.post("/feedback")
async def feedback(req: ChatRequest):
    try:
        resume_text = resume_store.get(req.session_id, "")
        chat_log = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in req.history]
        )

        prompt = (
            "You're an expert AI interviewer. Based on the following interview transcript and resume (if any), "
            "provide two things:\n\n"
            "1. A score from 1 to 10 based on the candidate's overall performance.\n"
            "2. A 3–5 sentence feedback highlighting strengths and areas for improvement.\n\n"
        )

        if resume_text:
            prompt += f"Resume:\n{resume_text}\n\n"

        prompt += f"Interview Transcript:\n{chat_log}\n\n"

        prompt += (
            "Respond in the following JSON format:\n"
            '{"score": <number from 1 to 10>, "feedback": "<text>"}'
        )

        messages = [{"role": "system", "content": prompt}]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4
        )

        import json
        parsed = json.loads(response.choices[0].message.content)

        return parsed

    except Exception as e:
        return {"error": str(e)}



#comment 
