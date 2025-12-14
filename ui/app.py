import streamlit as st
import requests
import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())



API_URL = "http://localhost:8000/ask"
UPLOAD_URL = "http://localhost:8000/upload"


st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Research Paper Assistant")
st.caption("Ask questions about your research papers")

st.subheader("📄 Upload a Research Paper (PDF)")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Upload PDF"):
        with st.spinner("Uploading and indexing PDF..."):
            response = requests.post(
                UPLOAD_URL,
                files={"file": uploaded_file},
                timeout=120
            )

        if response.status_code == 200:
            data = response.json()
            st.success(
                f"✅ {data['file']} uploaded "
                f"({data['chunks_added']} chunks indexed)"
            )
        else:
            st.error("❌ Failed to upload PDF")



question = st.text_input(
    "Enter your question",
    placeholder="e.g. What is this paper about?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json={
                    "question": question,
                    "session_id": st.session_state.session_id
                },
                timeout=60
            )

        if response.status_code != 200:
            st.error("Error from backend.")
        else:
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            if data["sources"]:
                st.subheader("Sources")
                for src in data["sources"]:
                    st.write(f"- {src}")
