import streamlit as st
import sqlite3
from hashlib import sha256
import bcrypt
import requests
import os
from datetime import datetime
from typing import List
import tempfile
# Optional PDF extraction
try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False
# Retriever & text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -----------------------
# Config
# -----------------------
st.set_page_config(title="iDigitalPreneur Assistant", layout="wide")
APP_TITLE = "iDigitalPreneur Assistant"
DB_PATH = "users.db"  # created in app folder
# Z.ai key from Streamlit secrets (set in Streamlit Cloud)
ZAI_API_KEY = st.secrets.get("z_ai", {}).get("api_key", None)
ZAI_BASE_URL = st.secrets.get("z_ai", {}).get("base_url", "https://api.z.ai/paas/v4/chat/completions")
ZAI_MODEL = st.secrets.get("z_ai", {}).get("model", "glm-4.5")

# -----------------------
# Initialize session state variables
# -----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# DB helpers
# -----------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash BLOB,
        created_at TEXT
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        role TEXT,
        content TEXT,
        created_at TEXT
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        content TEXT,
        created_at TEXT
    );
    """)
    conn.commit()
    conn.close()

def register_user_db(username: str, password: str) -> bool:
    try:
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, pw_hash, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

def verify_user_db(username: str, password: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return False
        stored = row[0]
        return bcrypt.checkpw(password.encode("utf-8"), stored)
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def save_message_db(user_id: int, role: str, content: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, role, content, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving message: {str(e)}")

def save_upload_db(user_id:int, filename:str, content:str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO uploads (user_id, filename, content, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, filename, content, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving upload: {str(e)}")

def get_user_id(username: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        r = cur.fetchone()
        conn.close()
        return r[0] if r else None
    except Exception as e:
        st.error(f"Error getting user ID: {str(e)}")
        return None

# Initialize database
init_db()

# -----------------------
# Text utils + retriever
# -----------------------
def clean_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def extract_text_from_pdf_bytes(b: bytes) -> str:
    if not HAS_PYMUPDF:
        raise RuntimeError("pymupdf not available in environment")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(b)
        tf.flush()
        path = tf.name
    try:
        doc = fitz.open(path)
        pages = []
        for p in doc:
            pages.append(p.get_text("text"))
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF processing error: {str(e)}")
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

class Retriever:
    def __init__(self, chunks: List[str]):
        self.chunks = [clean_text(c) for c in chunks if c and len(c.strip()) > 0]
        if not self.chunks:
            self.vectorizer = None
            self.doc_vectors = None
            return
        try:
            self.vectorizer = TfidfVectorizer().fit(self.chunks)
            self.doc_vectors = self.vectorizer.transform(self.chunks)
        except Exception as e:
            st.error(f"Error initializing vectorizer: {str(e)}")
            self.vectorizer = None
            self.doc_vectors = None
            
    def query(self, q: str, top_k: int = 3):
        if not self.vectorizer or not self.doc_vectors or not self.chunks:
            return []
        try:
            qv = self.vectorizer.transform([q])
            sims = cosine_similarity(qv, self.doc_vectors).flatten()
            ids = sims.argsort()[::-1][:top_k]
            return [(int(i), float(sims[i]), self.chunks[i]) for i in ids if i < len(self.chunks)]
        except Exception as e:
            st.error(f"Error in retriever query: {str(e)}")
            return []

def chunk_text(text: str, chunk_size:int=900, overlap:int=150):
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# -----------------------
# z.ai client (flexible parsing)
# -----------------------
def zai_chat(system_prompt: str, messages: list, max_tokens:int=512, temperature:float=0.2):
    if not ZAI_API_KEY:
        raise RuntimeError("Z.ai API key not found in Streamlit secrets (section 'z_ai').")
    
    body = {
        "model": ZAI_MODEL,
        "messages": [],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if system_prompt:
        body["messages"].append({"role":"system", "content": system_prompt})
    body["messages"].extend(messages)
    
    headers = {
        "Authorization": f"Bearer {ZAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(ZAI_BASE_URL, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        
        # Try common response shapes
        if "choices" in j and j["choices"]:
            ch = j["choices"][0]
            if isinstance(ch.get("message"), dict):
                return ch["message"].get("content", "")
            if ch.get("text"):
                return ch.get("text", "")
        if "data" in j and j["data"]:
            d = j["data"][0]
            return d.get("content", "") or d.get("text", "") or str(d)
        return str(j)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Response parsing error: {str(e)}")

# -----------------------
# Streamlit UI
# -----------------------
st.title(APP_TITLE)
st.write("Welcome! Login or register on the left. Upload a PDF/TXT with details about iDigitalPreneur and the assistant will use it to answer questions.")

# Sidebar auth
st.sidebar.header("Account")
auth_mode = st.sidebar.selectbox("Action", ["Login", "Register", "Logout"])

if auth_mode == "Register":
    reg_user = st.sidebar.text_input("Choose a username", key="reg_user")
    reg_pwd = st.sidebar.text_input("Choose a password", type="password", key="reg_pwd")
    if st.sidebar.button("Create account"):
        if not reg_user or not reg_pwd:
            st.sidebar.error("Enter username and password")
        else:
            ok = register_user_db(reg_user, reg_pwd)
            if ok:
                st.sidebar.success("Account created — please login")
            else:
                st.sidebar.error("Username exists or error")

elif auth_mode == "Login":
    login_user = st.sidebar.text_input("Username", key="login_user")
    login_pwd = st.sidebar.text_input("Password", type="password", key="login_pwd")
    if st.sidebar.button("Login"):
        if verify_user_db(login_user, login_pwd):
            st.session_state.logged_in = True
            st.session_state.username = login_user
            st.sidebar.success(f"Logged in as {login_user}")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

elif auth_mode == "Logout":
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("Logged out")
        st.rerun()

# require login
if not st.session_state.logged_in:
    st.info("Please login to use the assistant")
    st.stop()

username = st.session_state.username
user_id = get_user_id(username)
st.sidebar.markdown(f"**Signed in:** {username}")

# Upload / knowledge base
st.header("Upload Knowledge Document (PDF or TXT)")
u = st.file_uploader("Upload a PDF or TXT file with product/course/faq info (optional). The assistant will use it as context.", type=["pdf","txt"])
retriever = None
uploaded_text_preview = None

if u is not None:
    try:
        if u.type == "application/pdf" or (u.name.lower().endswith(".pdf")):
            if not HAS_PYMUPDF:
                st.error("PDF extraction not available (pymupdf missing). Please upload TXT or enable pymupdf in requirements.")
            else:
                with st.spinner("Processing PDF..."):
                    raw = u.read()
                    text = extract_text_from_pdf_bytes(raw)
                    uploaded_text_preview = text[:2000]
        else:
            with st.spinner("Processing text file..."):
                raw = u.read().decode("utf-8", errors="ignore")
                text = raw
                uploaded_text_preview = text[:2000]
        
        # chunk & build retriever
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        if chunks:
            retriever = Retriever(chunks)
            save_upload_db(user_id, u.name, text)
            st.success(f"Document loaded: {u.name} — {len(chunks)} chunks created.")
            with st.expander("Preview (first 2000 chars)"):
                st.text_area("", uploaded_text_preview, height=200)
        else:
            st.warning("No content could be extracted from the file.")
    except Exception as e:
        st.error(f"Failed to process upload: {e}")

# Chat UI (history)
st.header("Chat with Assistant")

# render history
for item in st.session_state.history:
    role, txt = item
    if role == "user":
        st.markdown(f"**You:** {txt}")
    else:
        st.markdown(f"**Assistant:** {txt}")

col1, col2 = st.columns([4,1])
with col1:
    query = st.text_input("Your question about iDigitalPreneur (courses, pricing, webinar, outcomes):", key="query_input")
with col2:
    send = st.button("Send")

if send and query.strip():
    st.session_state.history.append(("user", query))
    save_message_db(user_id, "user", query)
    
    # Build prompt with retrieved context if available
    system_prompt = (
        "You are a friendly, persuasive, human-like sales assistant for iDigitalPreneur "
        "(https://idigitalpreneur.com). Use supplied context when relevant. Encourage users to join the webinar and clarify doubts honestly."
    )
    
    messages = []
    context_text = ""
    
    if retriever:
        with st.spinner("Searching knowledge base..."):
            top = retriever.query(query, top_k=3)
            context_pieces = []
            for idx, score, chunk in top:
                context_pieces.append(f"[Doc chunk score={score:.3f}]\n{chunk}")
            if context_pieces:
                context_text = "\n\n".join(context_pieces)
                messages.append({"role":"user", "content": f"Context:\n{context_text}\n\nUser question: {query}"})
    
    if not context_text:
        messages.append({"role":"user", "content": query})
    
    with st.spinner("Contacting model..."):
        try:
            reply = zai_chat(system_prompt, messages, max_tokens=512, temperature=0.2)
        except Exception as e:
            reply = f"Model error: {e}"
    
    st.session_state.history.append(("assistant", reply))
    save_message_db(user_id, "assistant", reply)
    st.rerun()

# small footer
st.markdown("---")
st.markdown("Tip: Upload a clean FAQ or product doc (TXT or PDF) for best grounded answers. If PDF upload fails on Streamlit Cloud, upload TXT instead.")
