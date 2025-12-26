import streamlit as st
import speech_recognition as sr
import tempfile, os, re, threading, queue
import numpy as np
import faiss
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import PyPDFLoader
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

st.set_page_config(layout="wide")
st.title("📄 Azure RAG Q&A (Voice + Text Input)")

# Config
AZURE_CONNECTION_STRING = st.secrets.get("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "kalyanpdf"
API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Thread-safe queue
result_queue = queue.Queue()

# Initialize session_state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Build FAISS index
@st.cache_resource
def build_index():
    container = BlobServiceClient.from_connection_string(
        AZURE_CONNECTION_STRING
    ).get_container_client(CONTAINER_NAME)
    texts = []
    for blob in container.list_blobs():
        if not blob.name.lower().endswith((".pdf", ".docx", ".txt")):
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(blob.name)[1])
        tmp.write(container.get_blob_client(blob.name).download_blob().readall())
        tmp.close()
        if blob.name.endswith(".pdf"):
            text = "\n".join(p.page_content for p in PyPDFLoader(tmp.name).load())
        elif blob.name.endswith(".docx"):
            doc = DocxDocument(tmp.name)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            with open(tmp.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        os.remove(tmp.name)
        texts.append(re.sub(r"\s+", " ", text))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [c for t in texts for c in splitter.split_text(t)]
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vectors = np.array(embedder.embed_documents(chunks), dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, chunks, embedder

index, chunks, embedder = build_index()

# Helpers
def build_prompt(context, question):
    return f"""
Use ONLY the context to answer.

Context:
{context}

Question: {question}
Answer:
"""

def safe_post(url, headers, json_data):
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session.post(url, headers=headers, json=json_data, timeout=30)

def generate_answer(context, question):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    data = {
        "contents": [{"parts": [{"text": build_prompt(context, question)}]}]
    }
    res = safe_post(f"{url}?key={API_KEY}", {"Content-Type": "application/json"}, data)
    return res.json()["candidates"][0]["content"]["parts"][0]["text"]

def embed_query(q):
    return np.array(embedder.embed_query(q), dtype="float32")

def retrieve_context(question, top_k=3):
    q_emb = embed_query(question)
    _, I = index.search(q_emb.reshape(1, -1), top_k)
    context = "\n\n".join([chunks[i] for i in I[0]])
    return context

# Background thread function
def process_question(question_text):
    context = retrieve_context(question_text)
    answer = generate_answer(context, question_text)
    result_queue.put({"question": question_text, "answer": answer})

# Layout: text + voice
col1, col2 = st.columns([4,1])
with col1:
    question_text = st.text_input("Type your question here")
with col2:
    voice_clicked = st.button("🎤")

# Start threads on click
if st.button("Generate Answer") and question_text:
    threading.Thread(target=process_question, args=(question_text,), daemon=True).start()
    st.info("Processing your question...")

if voice_clicked:
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            voice_text = recognizer.recognize_google(audio)
            if voice_text:
                st.write("You asked:", voice_text)
                threading.Thread(target=process_question, args=(voice_text,), daemon=True).start()
                st.info("Processing your question...")
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

# =====================
# Placeholder for dynamic update
# =====================
placeholder = st.empty()

# Poll the queue **every 0.5s** to update answers
def poll_queue():
    while not result_queue.empty():
        result = result_queue.get_nowait()
        st.session_state.qa_history.append(result)

poll_queue()

# Display all Q&A
for item in st.session_state.qa_history:
    placeholder.markdown(f"**Question:** {item['question']}")
    placeholder.markdown(f"**Answer:** {item['answer']}")
    placeholder.markdown("---")
