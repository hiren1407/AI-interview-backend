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
import time
import asyncio
import httpx
import aiofiles


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
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input="Give me relevant interview questions"
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter={"session_id": session_id}
    )

    resume_context = " ".join([r["metadata"]["text"] for r in results["matches"] if r["metadata"]["type"] == "resume"])
    jd_context = " ".join([r["metadata"]["text"] for r in results["matches"] if r["metadata"]["type"] == "jd"])

    return resume_context, jd_context or None

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

tts_semaphore = asyncio.Semaphore(5)
async def get_deepgram_audio_base64(text: str):
    if not text.strip():
        return None

    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    url = "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = { "text": text.strip() }

    async with tts_semaphore:
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    print("TTS Error:", response.text)
                    return None
                return base64.b64encode(response.content).decode("utf-8")
            except Exception as e:
                print("TTS Exception:", e)
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

        # Extract candidate's name for personalization
        candidate_name = extract_name_from_resume(text)
        if candidate_name:
            name_store[session_id] = candidate_name
            print(f"Extracted name for session {session_id}: {candidate_name}")

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
    import time
    start = time.time()

    # Save uploaded audio file temporarily in a thread (blocking)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
        tmp_path = tmp.name

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with aiofiles.open(tmp_path, "rb") as f:
                audio_bytes = await f.read()

            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": file.content_type,
            }
            params = {
                "punctuate": "true",
                "model": "nova-3",
                "language": "en",
            }

            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=params,
                content=audio_bytes
            )

        if response.status_code != 200:
            try:
                return {"error": response.json().get("error", "Transcription failed")}
            except Exception:
                return {"error": "Transcription failed and response was not JSON"}

        result = response.json()
        transcript = result["results"]["channels"][0]["alternatives"][0].get("transcript", "")

        print("Transcription took:", time.time() - start)
        return {"transcript": transcript}

    except Exception as e:
        return {"error": str(e)}

    finally:
        os.remove(tmp_path)








@app.post("/respond")
async def respond(req: ChatRequest):
    total = time.time()
    try:
        
        if req.transcript.strip() == "":
            candidate_name = name_store.get(req.session_id, "")
            name_greeting = f"Hello {candidate_name}! " if candidate_name else ""
            welcome_text = (
                f"{name_greeting}Welcome! Let's begin your interview on {req.topic}. "
                "Let me know when you're ready to start."
            )
            audio_base64 = await get_deepgram_audio_base64(welcome_text)
            return {
                "reply": welcome_text,
                "audio_base64": audio_base64
            }

        # Check what data is available for this session
        has_resume = await asyncio.to_thread(check_resume_available, req.session_id)
        has_jd = await asyncio.to_thread(check_jd_available, req.session_id)
        
        # Get candidate's name if available
        candidate_name = name_store.get(req.session_id, "")
        name_prefix = f"{candidate_name}, " if candidate_name else ""
        
        print(f"Session {req.session_id}: has_resume={has_resume}, has_jd={has_jd}, topic={req.topic}, name={candidate_name}")

        # Construct prompt
        if req.topic == "Resume Based Questions":
            if not has_resume:
                error_message = "I notice you've selected 'Resume Based Questions' but no resume has been uploaded for this session. Please upload your resume first, or choose a different interview topic."
                audio_base64 = await get_deepgram_audio_base64(error_message)
                return {
                    "reply": error_message,
                    "audio_base64": audio_base64,
                    "error_type": "missing_resume"
                }
                
            resume_context, jd_context = await asyncio.to_thread(pinecone_search, req.session_id)
            prompt = (
                "You are an AI interviewer conducting a mock interview. You are the interviewer, not an advisor.\n\n"
                "You only have access to the candidate's resume — there is no job description.\n\n"
                "Ask questions that:\n"
                "- Dive deep into the resume experiences\n"
                "- Evaluate technical skills, leadership, and accomplishments\n"
                "- Never ask for job position clarification (since no JD is provided)\n\n"
                "CRITICAL: You are actively interviewing the candidate. Do NOT provide meta-advice or suggestions about how to handle responses. "
                "Instead, respond directly as an interviewer would.\n\n"
                "STAY FOCUSED: This is a RESUME-BASED interview. If the candidate mentions technologies or topics not in their resume or unrelated to the discussion, "
                "acknowledge briefly but redirect back to their resume experiences. For example: "
                "'That's interesting, but let's focus on your resume. Can you tell me more about [specific resume item]?' "
                "Do NOT start exploring unrelated technologies they mention if they're not relevant to the resume discussion.\n\n"
                "If a candidate gives short responses like 'no', 'nope', or seems disengaged:\n"
                "- Gently probe for more details\n"
                "- Rephrase the question\n"
                "- Move to a related topic\n"
                "- Keep the conversation flowing naturally\n\n"
                "RESPONSE FORMAT - Always follow this structure:\n"
                "1. One brief feedback sentence (optional)\n"
                "2. ONE single interview question\n"
                "3. Stop there - do not ask multiple questions or provide options\n\n"
                "NEVER ask multiple questions like 'What was your role? How did you handle challenges? What did you learn?'\n"
                "Instead ask: 'What was your role in that project?'\n\n"
                f"- The candidate's name is {candidate_name}. Use it naturally and sparingly.\n\n" if candidate_name else "- Keep questions professional and engaging\n\n"
                f"### Candidate Resume:\n{resume_context}"
            )
        elif req.topic == "Resume + JD Based Questions":
            if not has_resume:
                error_message = "I notice you've selected 'Resume + JD Based Questions' but no resume has been uploaded for this session. Please upload your resume first."
                audio_base64 = await get_deepgram_audio_base64(error_message)
                return {
                    "reply": error_message,
                    "audio_base64": audio_base64,
                    "error_type": "missing_resume"
                }
            
            if not has_jd:
                error_message = "I notice you've selected 'Resume + JD Based Questions' but no job description has been uploaded for this session. Please upload the job description first, or choose 'Resume Based Questions' instead."
                audio_base64 = await get_deepgram_audio_base64(error_message)
                return {
                    "reply": error_message,
                    "audio_base64": audio_base64,
                    "error_type": "missing_jd"
                }
                
            resume_context, jd_context = await asyncio.to_thread(pinecone_search, req.session_id)
            prompt = (
                "You are an AI interviewer conducting a mock interview. You are the interviewer, not an advisor.\n\n"
                "The candidate has applied for a job described below. You also have access to their resume.\n\n"
                "Ask questions that:\n"
                "- Match the job description\n"
                "- Dig into relevant past experiences\n"
                "- Never ask generic questions like which position?\n\n"
                "CRITICAL: You are actively interviewing the candidate. Do NOT provide meta-advice or suggestions about how to handle responses. "
                "Instead, respond directly as an interviewer would.\n\n"
                "STAY FOCUSED: This interview is for the specific job described below. If the candidate mentions technologies or experiences "
                "not related to the job requirements or their relevant resume experiences, acknowledge briefly but redirect back to job-relevant topics. "
                "For example: 'That's interesting, but for this role, I'd like to focus on [job-relevant skill]. Can you tell me about...?' "
                "Do NOT start exploring unrelated technologies or experiences if they don't match the job requirements.\n\n"
                "If a candidate gives short responses like 'no', 'nope', or seems disengaged:\n"
                "- Gently probe for more details\n"
                "- Rephrase the question\n"
                "- Move to a related topic\n"
                "- Keep the conversation flowing naturally\n\n"
                "RESPONSE FORMAT - Always follow this structure:\n"
                "1. One brief feedback sentence (optional)\n"
                "2. ONE single interview question\n"
                "3. Stop there - do not ask multiple questions or provide options\n\n"
                "NEVER ask multiple questions like 'What was your role? How did you handle challenges? What did you learn?'\n"
                "Instead ask: 'What was your role in that project?'\n\n"
                f"- The candidate's name is {candidate_name}. Use it naturally and sparingly.\n\n" if candidate_name else "- Keep questions professional and engaging\n\n"
                f"### Job Description:\n{jd_context}\n\n"
                f"### Candidate Resume:\n{resume_context}"
            )
        else:
            # For other topics, check if resume is available and use it for context
            resume_context, jd_context = await asyncio.to_thread(pinecone_search, req.session_id)
            
            base_prompt = (
                f"You are a friendly and professional AI interviewer for the topic: {req.topic}. You are the interviewer, not an advisor. "
                "Always respond in English. "
                "CRITICAL: Do NOT provide meta-advice or suggestions about how to handle responses. "
                "Instead, respond directly as an interviewer would. "
                f"STAY FOCUSED: This interview is specifically about {req.topic}. If the candidate mentions unrelated technologies or topics, "
                "acknowledge briefly but ALWAYS redirect back to the main topic. For example: "
                "'That's interesting, but let's focus on [main topic]. Can you tell me about...?' "
                "Do NOT change the interview focus or start asking about the unrelated technology they mentioned. "
                "If a candidate gives short responses like 'no', 'nope', or seems disengaged: "
                "gently probe for more details, rephrase the question, or move to a related topic naturally. "
                "Avoid summarizing the entire interview or ending the session unless explicitly asked. "
                "Do not say 'Do you have any questions?' or 'Thank you for your time' unless the user clearly signals the interview is over. "
                "RESPONSE FORMAT - Always follow this structure: "
                "1. One brief feedback sentence (optional) "
                "2. ONE single interview question RELATED TO THE MAIN TOPIC "
                "3. Stop there - do not ask multiple questions or provide options. "
                "NEVER ask multiple questions like 'What was your experience? How did you handle it? What did you learn?' "
                "Instead ask: 'What was your experience with that technology?'"
                f" The candidate's name is {candidate_name}. Use it naturally and sparingly." if candidate_name else ""
            )
            
            # Add resume context if available
            if resume_context and resume_context.strip():
                prompt = (
                    base_prompt + "\n\n"
                    "You have access to the candidate's resume. Use this information to ask more targeted questions "
                    "and provide relevant feedback based on their background and experience.\n\n"
                    f"However, keep the focus on {req.topic}. If they mention technologies not related to {req.topic}, "
                    "acknowledge briefly and redirect: 'That's good to know, but for this {req.topic} interview, let's focus on...'\n\n"
                    "RESPONSE FORMAT - Always follow this structure:\n"
                    "1. One brief feedback sentence (optional)\n"
                    "2. ONE single interview question RELATED TO THE MAIN TOPIC\n"
                    "3. Stop there - do not ask multiple questions or provide options\n\n"
                    "NEVER ask multiple questions like 'What was your experience? How did you handle it? What did you learn?'\n"
                    "Instead ask: 'What was your experience with that technology?'\n\n"
                    f"### Candidate Resume:\n{resume_context}"
                )
            else:
                prompt = (
                    base_prompt + "\n\n"
                    f"You do not have access to the candidate's resume, so focus on their general knowledge and experience in {req.topic}. "
                    "Ask one question at a time and build the conversation naturally based on their responses.\n\n"
                    f"STAY STRICTLY ON TOPIC: This interview is about {req.topic}. If the candidate mentions unrelated technologies "
                    f"(like talking about Java in a Python interview), acknowledge briefly but redirect: "
                    f"'That's interesting, but let's keep our focus on {req.topic}. Can you tell me about...?'\n\n"
                    "RESPONSE FORMAT - Always follow this structure:\n"
                    "1. One brief feedback sentence (optional)\n"
                    f"2. ONE single interview question ABOUT {req.topic}\n"
                    "3. Stop there - do not ask multiple questions or provide numbered lists\n\n"
                    "NEVER provide multiple questions like:\n"
                    "'1. Basic Concepts: Can you explain...?\n"
                    "2. Control Flow: How does...?\n"
                    "3. Functions: What is...?'\n\n"
                    f"Instead ask ONE question like: 'Can you explain [specific {req.topic} concept]?'\n\n"
                    "Do NOT provide numbered lists or multiple choice options. Act like a human interviewer having a conversation."
                )

        # Construct messages for OpenAI
        messages = [
            {"role": "system", "content": prompt},
            *req.history,
            {"role": "user", "content": f"The candidate answered: \"{req.transcript}\""}
        ]
        start = time.time()
        # Run OpenAI call in background thread (blocking)
        openai_response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4
        )
        reply_text = openai_response.choices[0].message.content.strip()
        print("GPT took", time.time() - start)

        # Get audio (optional, in background thread too)
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
        start = time.time()
        audio_base64 = await get_deepgram_audio_base64(reply_text)
        print("Deepgram tts took", time.time() - start)
        
        print("Total res time", time.time() - total)
        return {
            "reply": reply_text,
            "audio_base64": audio_base64
        }

    except Exception as e:
        return {"error": str(e)}


    
@app.post("/feedback")
async def feedback(req: ChatRequest):
    try:
        # Get resume and JD context from Pinecone for better feedback
        resume_context, jd_context = await asyncio.to_thread(pinecone_search, req.session_id)
        
        # Format the conversation history better
        chat_log = []
        for i, message in enumerate(req.history):
            if message['role'] == 'user':
                chat_log.append(f"Candidate Response {i//2 + 1}: {message['content']}")
            elif message['role'] == 'assistant':
                chat_log.append(f"Interviewer Question {i//2 + 1}: {message['content']}")
        
        conversation_text = "\n\n".join(chat_log)

        # Build comprehensive prompt for better feedback
        prompt = (
            "You are an expert interview evaluation specialist. Analyze this interview transcript carefully and provide detailed, accurate feedback.\n\n"
            "EVALUATION CRITERIA:\n"
            "- Technical Knowledge & Skills (25%)\n"
            "- Communication & Clarity (20%)\n"
            "- Problem-Solving Approach (20%)\n"
            "- Relevant Experience Discussion (15%)\n"
            "- Professionalism & Engagement (10%)\n"
            "- Question Handling & Follow-ups (10%)\n\n"
            
            "SCORING GUIDELINES:\n"
            "- 9-10: Exceptional performance, ready for senior roles\n"
            "- 7-8: Strong performance, good candidate\n"
            "- 5-6: Average performance, some concerns\n"
            "- 3-4: Below average, significant gaps\n"
            "- 1-2: Poor performance, major deficiencies\n\n"
        )

        # Add context if available
        if resume_context and resume_context.strip():
            prompt += f"CANDIDATE'S RESUME CONTEXT:\n{resume_context}\n\n"
        
        if jd_context and jd_context.strip():
            prompt += f"JOB REQUIREMENTS:\n{jd_context}\n\n"
        
        if req.topic:
            prompt += f"INTERVIEW TOPIC: {req.topic}\n\n"

        prompt += f"INTERVIEW TRANSCRIPT:\n{conversation_text}\n\n"

        prompt += (
            "INSTRUCTIONS:\n"
            "1. Analyze each response for depth, accuracy, and relevance\n"
            "2. Consider the candidate's communication style and clarity\n"
            "3. Evaluate how well they handled follow-up questions\n"
            "4. Compare their responses to their resume (if available)\n"
            "5. Assess technical accuracy and problem-solving approach\n\n"
            
            "Provide your evaluation in the following JSON format:\n"
            "{\n"
            '  "score": <integer from 1-10>,\n'
            '  "feedback": "<detailed 4-6 sentence analysis covering strengths, weaknesses, and specific recommendations>",\n'
            '  "technical_score": <integer from 1-10>,\n'
            '  "communication_score": <integer from 1-10>,\n'
            '  "strengths": ["<strength1>", "<strength2>"],\n'
            '  "areas_for_improvement": ["<area1>", "<area2>"],\n'
            '  "specific_examples": "<reference specific parts of the conversation>"\n'
            "}\n\n"
            "Be specific, constructive, and avoid generic statements. Reference actual parts of the conversation."
        )

        messages = [{"role": "system", "content": prompt}]

        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 for better analysis
            messages=messages,
            temperature=0.2  # Lower temperature for more consistent feedback
        )

        import json
        try:
            parsed = json.loads(response.choices[0].message.content)
            
            # Validate the response structure
            required_fields = ["score", "feedback", "technical_score", "communication_score", "strengths", "areas_for_improvement"]
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            return parsed
            
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
            
            # Try to extract basic score and feedback
            content = response.choices[0].message.content
            return {
                "score": 5,
                "feedback": content,
                "technical_score": 5,
                "communication_score": 5,
                "strengths": ["Participated in the interview"],
                "areas_for_improvement": ["Need more detailed analysis"],
                "specific_examples": "Analysis could not be parsed properly",
                "error": "JSON parsing failed, showing raw response"
            }

    except Exception as e:
        print(f"Feedback endpoint error: {e}")
        return {"error": str(e)}



#comment

def check_resume_available(session_id: str) -> bool:
    """Check if resume data is available for the given session_id"""
    try:
        # Check in pinecone index
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input="resume content"
        ).data[0].embedding

        results = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            filter={"session_id": session_id, "type": "resume"}
        )
        
        return len(results["matches"]) > 0
    except Exception as e:
        print(f"Error checking resume availability: {e}")
        return False

def check_jd_available(session_id: str) -> bool:
    """Check if job description data is available for the given session_id"""
    try:
        # Check in pinecone index
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input="job description content"
        ).data[0].embedding

        results = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            filter={"session_id": session_id, "type": "jd"}
        )
        
        return len(results["matches"]) > 0
    except Exception as e:
        print(f"Error checking JD availability: {e}")
        return False

@app.get("/session-data/{session_id}")
async def get_session_data(session_id: str):
    """Get information about what data is available for a session"""
    try:
        has_resume = await asyncio.to_thread(check_resume_available, session_id)
        has_jd = await asyncio.to_thread(check_jd_available, session_id)
        
        return {
            "session_id": session_id,
            "has_resume": has_resume,
            "has_jd": has_jd
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/session-data/{session_id}")
async def delete_session_data(session_id: str):
    """Delete all data for a specific session"""
    try:
        # Remove from in-memory stores
        if session_id in resume_store:
            del resume_store[session_id]
        if session_id in name_store:
            del name_store[session_id]
        
        # Delete from Pinecone index
        # Note: Pinecone delete by metadata filter
        try:
            # Get all vectors for this session
            query_embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input="dummy query"
            ).data[0].embedding
            
            results = index.query(
                vector=query_embedding,
                top_k=1000,  # Get all possible matches
                include_metadata=True,
                filter={"session_id": session_id}
            )
            
            # Delete all found vectors
            if results["matches"]:
                ids_to_delete = [match["id"] for match in results["matches"]]
                index.delete(ids=ids_to_delete)
                
        except Exception as e:
            print(f"Error deleting from Pinecone: {e}")
        
        return {
            "success": True,
            "message": f"All data for session {session_id} has been deleted"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_name_from_resume(resume_text: str) -> str:
    """Extract candidate's name from resume text using OpenAI"""
    try:
        if not resume_text or not resume_text.strip():
            return ""
            
        name_extraction_prompt = (
            "Extract the candidate's name from this resume text. "
            "Return only the first name or first and last name, nothing else. "
            "If no clear name is found, return an empty string.\n\n"
            f"Resume text:\n{resume_text[:1000]}"  # First 1000 chars to avoid token limits
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": name_extraction_prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        name = response.choices[0].message.content.strip()
        
        # Basic validation - name should be reasonable length and contain only letters/spaces
        if len(name) > 50 or not name.replace(" ", "").replace("-", "").isalpha():
            return ""
            
        return name
        
    except Exception as e:
        print(f"Error extracting name: {e}")
        return ""

# Store extracted names by session_id
name_store = {}
