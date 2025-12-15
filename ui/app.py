# ui/app.py
import streamlit as st
import requests
import uuid
import difflib
import time

# -----------------------------
# Page config (MUST be first)
# -----------------------------
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide"
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
PAPER_TEXT_URL = f"{API_BASE}/paper_text"

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📚 Uploaded Papers")

    role = st.radio("Role", ["Student", "Researcher", "Reviewer"])
    st.markdown("---")

    # Fetch uploaded papers (safe)
    papers = []
    try:
        resp = requests.get(PAPERS_URL, timeout=5)
        if resp.status_code == 200:
            papers = resp.json().get("papers", [])
    except Exception:
        # backend unreachable — show helpful text
        st.warning("Backend may be down — start the API server (uvicorn).")

    if papers:
        for p in papers:
            st.write("•", p)
    else:
        st.info("No papers uploaded yet")

    st.markdown("---")
    st.subheader("➕ Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file and st.button("Upload & Index"):
        with st.spinner("Uploading and indexing..."):
            try:
                r = requests.post(UPLOAD_URL, files={"file": uploaded_file}, timeout=120)
                if r.status_code == 200:
                    st.success("Uploaded successfully ✅")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Upload failed ({r.status_code}). Check backend logs.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown("---")
    mode = st.selectbox("Mode", ["Chat", "Summarize", "Compare"])

# -----------------------------
# Main area
# -----------------------------
st.title("📄 Research Paper Assistant")
st.caption("Chat, summarize, and compare research papers")

# =====================================================
# CHAT MODE
# =====================================================
if mode == "Chat":
    col1, col2 = st.columns([3, 1])

    with col1:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    if msg.get("sources"):
                        st.markdown("**Sources:**")
                        for s in msg["sources"]:
                            st.write("-", s)

        question = st.chat_input("Ask a question about the papers...")

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            answer = "⏳ Requesting answer..."
            sources = []
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        ASK_URL,
                        json={
                            "question": question,
                            "session_id": st.session_state.session_id,
                            "role": role.lower()
                        },
                        timeout=60
                    )
                    if r.status_code == 200:
                        data = r.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])
                    else:
                        answer = f"❌ Backend returned status {r.status_code}"
                except requests.exceptions.ConnectionError:
                    answer = "❌ Could not connect to backend. Is uvicorn running?"
                except Exception as e:
                    answer = f"❌ Request error: {e}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            # Re-render to show new assistant message
            st.rerun()

    with col2:
        st.subheader("Controls")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

# =====================================================
# SUMMARIZE MODE
# =====================================================
elif mode == "Summarize":
    st.subheader("📝 One-Click Paper Summary")

    if st.button("Summarize Papers"):
        with st.spinner("Summarizing..."):
            try:
                r = requests.post(SUMMARY_URL, params={"role": role.lower()}, timeout=120)
                if r.status_code == 200:
                    st.subheader("Summary")
                    st.write(r.json().get("summary", ""))
                else:
                    st.error(f"Summarize failed ({r.status_code}). Check backend.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Start uvicorn.")
            except Exception as e:
                st.error(f"Request failed: {e}")

# =====================================================
# COMPARE MODE
# =====================================================
elif mode == "Compare":
    st.subheader("🔍 Compare Two Papers")

    # Refresh papers list safely
    try:
        resp = requests.get(PAPERS_URL, timeout=5)
        if resp.status_code == 200:
            papers = resp.json().get("papers", [])
        else:
            papers = []
    except Exception:
        papers = []

    if len(papers) < 2:
        st.info("Upload at least two papers to compare")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            paper_a = st.selectbox("Paper A", papers)
        with col_b:
            # pick a different default if possible
            idx = 1 if len(papers) > 1 else 0
            paper_b = st.selectbox("Paper B", papers, index=idx)

        if st.button("Compare"):
            with st.spinner("Comparing..."):
                try:
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
                        st.write(r.json().get("comparison", ""))
                    else:
                        st.error(f"Compare failed ({r.status_code}).")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend. Start uvicorn.")
                except Exception as e:
                    st.error(f"Request failed: {e}")

            # -------- DIFF VIEW (uses /paper_text) --------
            try:
                r1 = requests.get(PAPER_TEXT_URL, params={"name": paper_a}, timeout=30)
                r2 = requests.get(PAPER_TEXT_URL, params={"name": paper_b}, timeout=30)
                if r1.status_code == 200 and r2.status_code == 200:
                    t1 = r1.json().get("text", "")
                    t2 = r2.json().get("text", "")
                else:
                    t1 = t2 = ""
            except Exception:
                t1 = t2 = ""

            if t1 and t2:
                st.markdown("---")
                if st.checkbox("Show text diff (sampled)"):
                    diff = difflib.unified_diff(
                        t1.splitlines()[:500],
                        t2.splitlines()[:500],
                        fromfile=paper_a,
                        tofile=paper_b,
                        lineterm=""
                    )
                    diff_text = "\n".join(diff)
                    if diff_text.strip() == "":
                        st.info("No differences found in sampled text.")
                    else:
                        st.code(diff_text)
            else:
                st.info("No text available for diff (maybe /paper_text missing or backend unreachable).")

# Footer
st.markdown("---")
st.caption("Tip: Start the backend with `uvicorn api.app:app --reload` if you haven't already.")
