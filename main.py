from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import tempfile
import shutil
import fitz 
from fastapi.responses import StreamingResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import OpenAI
from dotenv import load_dotenv
import requests

# from elevenlabs.client import ElevenLabs
import io
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4

load_dotenv()
import base64

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# elevenlabs = ElevenLabs(
#   api_key=os.getenv("ELEVENLABS_API_KEY"),
# )
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
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

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")

# Create index if not exists
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Change region if needed
    )

# Connect to the index
index = pc.Index(index_name)


class ChatRequest(BaseModel):
    topic: str
    transcript: str
    history: list
    session_id: str
    include_jd: Optional[bool] = False

resume_store = {}

def pinecone_search(session_id: str):
    

    # Create query embedding (can customize this prompt later)
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input="Give me relevant interview questions"
    ).data[0].embedding

    resume_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={"session_id": session_id, "type": "resume"}
    )

    jd_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={"session_id": session_id, "type": "jd"}
    )

    resume_context = " ".join([r["metadata"]["text"] for r in resume_results["matches"]])
    jd_context = " ".join([r["metadata"]["text"] for r in jd_results["matches"]])

    return resume_context, jd_context if jd_results["matches"] else None

def get_elevenlabs_audio_base64(text):
    try:
        voice_id = "WTnybLRChAQj0OBHYZg4"  # Default voice, can be customized
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            return None

        audio_bytes = response.content
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print("ElevenLabs error:", e)
        return None


def get_deepgram_audio_base64(text: str, voice: str = "aura-asteria-en"):
    """
    Converts text to speech using Deepgram's TTS API and returns base64-encoded audio.
    """
    try:
        
        if not text or not text.strip():
            print("Deepgram TTS error: Empty text input.")
            return None

        DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        url = "https://api.deepgram.com/v1/speak"

        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "text": text.strip()
        }

        response = requests.post(url,json=payload, headers=headers )
        

        if response.status_code != 200:
            print("Deepgram TTS error:", response.json())
            return None

        return base64.b64encode(response.content).decode("utf-8")

    except Exception as e:
        print("Deepgram TTS exception:", e)
        return None

    

@app.post("/upload-jd")
async def upload_jd(text: str = Form(...), session_id: str = Form(...)):
    try:
        # Store JD text in memory if needed (optional)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embedding = embedding_response.data[0].embedding

            index.upsert([{
                "id": f"{session_id}_jd_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "session_id": session_id,
                    "type": "jd"
                }
            }])

        

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    

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

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embedding = embedding_response.data[0].embedding
            index.upsert([{
            "id": f"{session_id}_resume_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "session_id": session_id,
                "type": "resume"
            }
        }])
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), session_id: str = Form(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # with open(tmp_path, "rb") as audio_file:
        #     transcript = client.audio.transcriptions.create(
        #         model="whisper-1",
        #         file=audio_file
        #     )
        # return {"transcript": transcript.text}
        with open(tmp_path, 'rb') as audio_file:
            response = requests.post(
                url="https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": file.content_type  # e.g., audio/webm
                },
                params={
                    "punctuate": "true",
                    "model": "nova",  # optional, most accurate
                    "language": "en"
                },
                data=audio_file
            )

        result = response.json()

        if response.status_code != 200:
            return {"error": result.get("error", "Transcription failed")}

        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        return {"transcript": transcript}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)

@app.post("/respond")
async def respond(req: ChatRequest):
    try:
        
        resume_context, jd_context = pinecone_search(req.session_id)
        if req.topic == "Resume Based Questions":
            prompt = (
                "You are an AI interviewer conducting a mock interview.\n\n"
                "You only have access to the candidate's resume — there is no job description.\n\n"
                "Ask questions that:\n"
                "- Dive deep into the resume experiences\n"
                "- Evaluate technical skills, leadership, and accomplishments\n"
                "- Never ask for job position clarification (since no JD is provided)\n\n"
                "Always respond in English. After each candidate response:\n"
                "- Provide exactly **one** brief, constructive or encouraging sentence of feedback\n"
                "- Then ask **one** relevant and concise follow-up interview question\n"
                "- Do not ask multiple questions at once or give long explanations.\n\n"
                f"### Candidate Resume:\n{resume_context}"
            )
        elif req.topic == "Resume + JD Based Questions":
            prompt = (
                "You are an AI interviewer conducting a mock interview.\n\n"
                "The candidate has applied for a job described below. You also have access to their resume.\n\n"
                "Ask questions that:\n"
                "- Match the job description\n"
                "- Dig into relevant past experiences\n"
                "- Never ask generic questions like which position?\n\n"
                "Always respond in English. After each candidate response:\n"
                "- Provide exactly **one** brief, constructive or encouraging sentence of feedback\n"
                "- Then ask **one** relevant and concise follow-up interview question\n"
                "- Do not ask multiple questions at once or give long explanations.\n\n"
                f"### Job Description:\n{jd_context}\n\n"
                f"### Candidate Resume:\n{resume_context}"
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

        #audio_base64 = get_elevenlabs_audio_base64(reply_text)


        # Generate TTS audio for the reply_text
        # tts_response = client.audio.speech.create(
        #     model="gpt-4o-mini-tts",
        #     voice="onyx",
        #     input=reply_text
        # )
        # audio_bytes = tts_response.content

        # # Convert audio bytes to base64 string for JSON transport
        # audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        if reply_text.strip():
            audio_base64 = get_deepgram_audio_base64(reply_text)
        else:
            audio_base64 = None

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
            "You're an expert AI interviewer and you are very strict. Based on the following interview transcript and resume (if any), "
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
