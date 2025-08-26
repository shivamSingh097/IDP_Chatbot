# app.py
import streamlit as st
from dotenv import load_dotenv
import os, io, time
from db import init_db, get_user_by_email, create_user, save_message, save_upload
from auth import hash_password, check_password
from doc_utils import extract_text_from_pdf, chunk_text, Retriever, clean_text
from zai_client import chat_completion
from datetime import datetime

load_dotenv()
APP_NAME = os.getenv("APP_NAME", "iDigitalPreneur Assistant")
WEBINAR_LINK = os.getenv("WEBINAR_LINK", "https://idigitalpreneur.com")
OWNER_NAME = os.getenv("OWNER_NAME", "Kamlesh Thakur")

st.set_page_config(page_title=APP_NAME, layout="centered")
init_db()

######################
# Helpers
######################
def login_ui():
    st.sidebar.title("Login / Register")
    tab = st.sidebar.radio("Action", ["Login", "Register"])
    if tab == "Login":
        email = st.sidebar.text_input("Email")
        pwd = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user = get_user_by_email(email.strip().lower())
            if not user:
                st.sidebar.error("No user found")
            else:
                if check_password(pwd, user["password_hash"]):
                    st.session_state["user"] = {"id":user["id"], "email":user["email"], "name":user["name"]}
                    st.sidebar.success(f"Welcome back, {user['name']}")
                else:
                    st.sidebar.error("Invalid credentials")
    else:
        st.sidebar.subheader("Register")
        name = st.sidebar.text_input("Name", key="reg_name")
        email = st.sidebar.text_input("Email", key="reg_email")
        pwd = st.sidebar.text_input("Password", type="password", key="reg_pwd")
        if st.sidebar.button("Create account"):
            if not name or not email or not pwd:
                st.sidebar.error("Fill all fields")
            else:
                if get_user_by_email(email.strip().lower()):
                    st.sidebar.error("User already exists")
                else:
                    create_user(email.strip().lower(), hash_password(pwd), name.strip())
                    st.sidebar.success("Account created — please login")

def logout_button():
    if "user" in st.session_state:
        if st.sidebar.button("Logout"):
            del st.session_state["user"]
            st.experimental_rerun()

######################
# UI layout
######################
if "user" not in st.session_state:
    st.session_state["user"] = None

login_ui()
logout_button()

st.title(APP_NAME)
st.markdown(f"**Welcome!** I’m an assistant trained to answer questions about iDigitalPreneur and help users join the webinar. (Owner: {OWNER_NAME})")

if not st.session_state["user"]:
    st.info("Please login or register using the sidebar to use the chat.")
    st.stop()

user = st.session_state["user"]
st.success(f"Signed in as: {user['name']} ({user['email']})")

# Upload doc section
st.header("Upload a knowledge document (PDF or TXT)")
uploaded = st.file_uploader("Upload a PDF or TXT that contains in-depth info about iDigitalPreneur (optional). The assistant will use this doc to answer questions.", type=["pdf","txt"])
retriever = None
if uploaded:
    file_bytes = uploaded.read()
    filename = uploaded.name
    text = ""
    if filename.lower().endswith(".pdf"):
        # write to temp file and extract
        with open("tmp_upload.pdf","wb") as f:
            f.write(file_bytes)
        text = extract_text_from_pdf("tmp_upload.pdf")
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
    text = clean_text(text)
    if len(text.strip()) < 20:
        st.error("Document seems empty or couldn't extract text.")
    else:
        # chunk and build retriever
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        retriever = Retriever(chunks)
        st.success(f"Document loaded. {len(chunks)} chunks created.")
        # save upload in DB
        save_upload(user["id"], filename, text)
        st.info("Document saved to your uploads.")

# Conversation panel
st.header("Chat with the assistant")
chat_container = st.container()

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of tuples (role, text)

# Show history
with chat_container:
    for role, txt in st.session_state["history"]:
        if role == "user":
            st.markdown(f"**You:** {txt}")
        else:
            st.markdown(f"**Assistant:** {txt}")

# Input box
query = st.text_input("Ask anything about iDigitalPreneur, the webinar, pricing, curriculum, or course outcomes. Be specific.", key="query")
if st.button("Send") and query.strip():
    # Save user message
    st.session_state["history"].append(("user", query))
    save_message(user["id"], "user", query)

    # Build system prompt and context
    system_prompt = (
        "You are a friendly, persuasive, human-like sales assistant for iDigitalPreneur (https://idigitalpreneur.com). "
        "Answer in a conversational tone, solve doubts about courses, pricing, credibility, outcomes, and encourage the user to join the webinar or purchase. "
        "Use provided document excerpts when available; always be honest. If user asks for pricing/link, provide the webinar link: "
        f"{WEBINAR_LINK} . Keep responses concise (~120-200 words) and clearly explain next steps."
    )

    # Retrieval
    context_text = ""
    if retriever:
        top = retriever.query(query, top_k=3)
        # attach top chunks
        context_pieces = []
        for idx, score, chunk in top:
            context_pieces.append(f"[Doc chunk score={score:.3f}]\n{chunk}")
        if context_pieces:
            context_text = "\n\n".join(context_pieces)
    # Build messages list
    messages = []
    if context_text:
        messages.append({"role":"user", "content": f"Context:\n{context_text}\n\nUser question: {query}"})
    else:
        messages.append({"role":"user", "content": query})

    # Call model
    with st.spinner("Thinking..."):
        try:
            resp = chat_completion(system_prompt, messages, max_tokens=512, temperature=0.2)
            assistant_text = resp if isinstance(resp, str) else str(resp)
        except Exception as e:
            assistant_text = f"Sorry — model call failed: {e}"

    # Save assistant reply
    st.session_state["history"].append(("assistant", assistant_text))
    save_message(user["id"], "assistant", assistant_text)
    # Rerun to show updated conversation
    st.experimental_rerun()

# Small hint and logout
st.markdown("---")
st.markdown("**Tips:** Provide the assistant with the uploaded document to get answers grounded in your file. If something looks incorrect, ask follow-ups.")
st.markdown("If you want, upload a well-written guide/FAQ about iDigitalPreneur and the assistant will use it to answer customer questions.")
