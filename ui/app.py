import streamlit as st
import requests
import uuid

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="centered"
)

# -----------------------------
# Session setup
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# API endpoints
# -----------------------------
API_BASE = "http://127.0.0.1:8000"
ASK_URL = f"{API_BASE}/ask"
UPLOAD_URL = f"{API_BASE}/upload"
SUMMARY_URL = f"{API_BASE}/summarize"
COMPARE_URL = f"{API_BASE}/compare"
PAPERS_URL = f"{API_BASE}/papers"

# -----------------------------
# Header
# -----------------------------
st.title("📄 Research Paper Assistant")
st.caption("Chat with your research papers")

# -----------------------------
# Role toggle
# -----------------------------
role = st.selectbox(
    "Who is using this?",
    ["Student", "Researcher", "Reviewer"]
)

# -----------------------------
# Uploaded papers list
# -----------------------------
st.subheader("📚 Uploaded Papers")

try:
    res = requests.get(PAPERS_URL, timeout=10)
    papers = res.json().get("papers", [])
except:
    papers = []

if papers:
    for p in papers:
        st.write(f"• {p}")
else:
    st.info("No papers uploaded yet.")

# -----------------------------
# Upload PDF
# -----------------------------
st.subheader("📄 Upload a Research Paper")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file and st.button("Upload PDF"):
    with st.spinner("Uploading and indexing..."):
        r = requests.post(
            UPLOAD_URL,
            files={"file": uploaded_file},
            timeout=120
        )

    if r.status_code == 200:
        data = r.json()
        st.success(
            f"✅ {data['file']} uploaded "
            f"({data['chunks_added']} chunks)"
        )
        st.rerun()
    else:
        st.error("Upload failed")

# -----------------------------
# Chat history
# -----------------------------
st.subheader("💬 Conversation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input("Ask a question about your papers")

if question:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking..."):
        r = requests.post(
            ASK_URL,
            json={
                "question": question,
                "session_id": st.session_state.session_id,
                "role": role.lower()
            },
            timeout=60
        )

    if r.status_code != 200:
        answer = "❌ Error from backend."
        sources = []
    else:
        data = r.json()
        answer = data["answer"]
        sources = data["sources"]

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            st.caption("Sources:")
            for s in sources:
                st.write(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# -----------------------------
# Summarize
# -----------------------------
if st.button("📄 Summarize All Papers"):
    with st.spinner("Summarizing..."):
        r = requests.post(
            SUMMARY_URL,
            params={"role": role.lower()},
            timeout=120
        )

    if r.status_code == 200:
        st.subheader("📝 Summary")
        st.write(r.json()["summary"])

# -----------------------------
# Compare papers
# -----------------------------
st.subheader("🔍 Compare Two Papers")

paper_a = st.selectbox("Paper A", papers)
paper_b = st.selectbox("Paper B", papers)

if st.button("Compare"):
    if paper_a == paper_b:
        st.warning("Choose two different papers.")
    else:
        with st.spinner("Comparing..."):
            r = requests.post(
                COMPARE_URL,
                json={
                    "paper_a": paper_a,
                    "paper_b": paper_b,
                    "role": role.lower()
                },
                timeout=120
            )

        if r.status_code == 200:
            st.subheader("📊 Comparison")
            st.write(r.json()["comparison"])
