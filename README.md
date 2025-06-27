# AI Interview App – Backend

This FastAPI backend powers the AI Interview App. It handles:
- Resume upload and parsing
- Audio transcription via OpenAI Whisper
- Interview logic and AI chat (GPT)
- Audio response generation via ElevenLabs
- Final interview feedback and scoring

---

## 🚀 Tech Stack

- Python 3.9+
- FastAPI
- OpenAI (GPT-4o, Whisper)
- ElevenLabs (Text-to-Speech)
- PyMuPDF (PDF parsing)
- Uvicorn

---

## 🧪 Features

- Resume PDF parsing and session-specific storage
- Interview topic selection, including resume-based interviews
- GPT-powered chat with dynamic context
- Audio response generation using ElevenLabs with natural voices
- Feedback scoring from interview transcript

---

## 🔐 Environment Variables

Create a `.env` file in the root:

```env
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload
