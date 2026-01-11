# app.py
from src.rag_pipeline import CrediTrustRAG
import sys
from pathlib import Path
import streamlit as st
import textwrap

# Fix Python path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your RAG system

# Page config
st.set_page_config(
    page_title="CrediTrust Complaint Assistant",
    page_icon="💬",
    layout="centered"
)

st.title("CrediTrust Complaint Assistant")

st.markdown("""
Ask questions about customer complaints across **Credit Cards**, **Personal Loans**, **Savings Accounts**, and **Money Transfers**.
Answers are generated locally using AI from real customer narratives.
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    st.session_state.rag = None


def initialize_rag():
    if st.session_state.rag is None:
        with st.status("Initializing RAG system (this may take 5-15 minutes on first use)...") as status:
            st.write("Loading full vector store (1.37M chunks)...")
            st.session_state.rag = CrediTrustRAG(top_k=5)
            st.write("RAG system ready!")
            status.update(label="RAG system loaded!", state="complete")
    return st.session_state.rag


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Your question (e.g., Why are customers unhappy with Credit Cards?)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show processing
    with st.chat_message("assistant"):
        with st.spinner("Processing... (first question may take several minutes)"):
            try:
                rag_system = initialize_rag()
                answer, sources = rag_system.ask(prompt)

                # Format answer
                formatted_answer = ""
                for word in answer.split():
                    formatted_answer += word + " "
                    if len(formatted_answer.split()[-1]) > 80:
                        formatted_answer += "\n"

                st.markdown(formatted_answer)

                # Add sources
                sources_text = "\n\n**Sources Used (Top 3):**\n"
                for i, src in enumerate(sources[:3], 1):
                    sources_text += f"\n**Source {i}** — Product: {src['product_category']} | Complaint ID: {src['complaint_id']}\n"
                    sources_text += f"{textwrap.shorten(src['text_preview'], width=500, placeholder='...')}\n"

                st.markdown(sources_text)

                # Save to history
                full_response = formatted_answer + sources_text
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {str(e)}"})

# Clear button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.markdown("""
**Note**: The first question may take **5–15 minutes** to load the full dataset and AI model (Llama 3.2 via Ollama). 
After that, responses are fast (~10–30 seconds).
""")
