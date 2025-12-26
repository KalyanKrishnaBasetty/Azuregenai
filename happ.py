# pip install fastapi uvicorn azure-storage-blob langchain-google-genai
# pip install langchain_community PyPDF2 faiss-cpu numpy requests

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
import os, tempfile, requests, faiss, numpy as np
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

AZURE = os.getenv("AZURE_CONNECTION_STRING")
KEY = os.getenv("GOOGLE_API_KEY")
CONTAINER = "kalyanpdf"

app = FastAPI()

# ---------- BUILD RAG ----------
texts = []
container = BlobServiceClient.from_connection_string(AZURE).get_container_client(CONTAINER)

for b in container.list_blobs():
    if b.name.endswith(".pdf"):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(container.get_blob_client(b.name).download_blob().readall())
        f.close()
        texts += [p.page_content for p in PyPDFLoader(f.name).load()]
        os.remove(f.name)

splitter = RecursiveCharacterTextSplitter(800, 100)
chunks = splitter.split_text(" ".join(texts))

embed = GoogleGenerativeAIEmbeddings("models/embedding-001", google_api_key=KEY)
vecs = np.array(embed.embed_documents(chunks)).astype("float32")
index = faiss.IndexFlatL2(vecs.shape[1])
index.add(vecs)

# ---------- ANSWER ----------
def answer(q):
    try:
        _, i = index.search(np.array(embed.embed_query(q)).reshape(1,-1), 3)
        ctx = "\n".join(chunks[x] for x in i[0])

        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={KEY}",
            json={"contents":[{"parts":[{"text": f"Context:\n{ctx}\n\nQ:{q}\nA:"}]}]},
            timeout=20
        )

        data = r.json()
        return data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","No answer generated")

    except Exception as e:
        return f"Error: {e}"

# ---------- FRONTEND ----------
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <h2>Azure RAG Q&A</h2>
    <textarea id=q rows=4 cols=60 placeholder="Ask something"></textarea><br><br>
    <button onclick=ask()>Ask</button>
    <pre id=a></pre>

    <script>
    async function ask(){
      a.innerText="Thinking...";
      let r = await fetch('/ask?q=' + encodeURIComponent(q.value));
      a.innerText = await r.text();
    }
    </script>
    """

# ---------- API ----------
@app.get("/ask", response_class=PlainTextResponse)
def ask(q:str):
    return answer(q)
