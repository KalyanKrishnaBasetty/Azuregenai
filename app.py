# pip install fastapi uvicorn requests

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import requests, os

app = FastAPI()
API_KEY = os.getenv("GOOGLE_API_KEY")

@app.get("/ask", response_class=PlainTextResponse)
def ask(q: str):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}",
        json={"contents":[{"parts":[{"text": q}]}]},
        timeout=20
    )
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
# pip install fastapi uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

app = FastAPI()

# ✅ VERY IMPORTANT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask", response_class=PlainTextResponse)
def ask(q: str):
    return "Backend is working. Your question was: " + q
