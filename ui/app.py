import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Research Paper Assistant")
st.caption("Ask questions about your research papers")

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
                json={"question": question},
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
