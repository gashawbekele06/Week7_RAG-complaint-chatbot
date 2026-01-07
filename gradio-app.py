# app.py
import sys
from pathlib import Path
import gradio as gr
import textwrap
import time

# Fix Python path for imports
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root added to Python path: {PROJECT_ROOT}")

# Import your RAG system
from src.rag_pipeline import CrediTrustRAG

# Global RAG instance (lazy initialization)
rag = None

def initialize_rag():
    global rag
    if rag is None:
        print("Initializing CrediTrust RAG Chatbot (this may take 5-15 minutes on first use)...")
        rag = CrediTrustRAG(top_k=5)
        print("RAG system ready!")
    return rag

def chat(message, history):
    if not message.strip():
        return history

    # Immediate feedback
    history.append((message, "Processing... (first question may take several minutes)"))

    try:
        rag_system = initialize_rag()
        answer, sources = rag_system.ask(message)

        # Format answer with line breaks
        formatted_answer = ""
        words = answer.split()
        for i, word in enumerate(words):
            formatted_answer += word + " "
            if (i + 1) % 12 == 0:  # Line break every ~12 words
                formatted_answer += "\n"
            # Simulate streaming
            time.sleep(0.03)
            yield history[:-1] + [(message, formatted_answer)]

        # Final answer
        history[-1] = (message, formatted_answer)

        # Add sources for transparency
        sources_text = "\n\n**Sources Used (Top 3):**\n"
        for i, src in enumerate(sources[:3], 1):
            sources_text += f"\n**Source {i}** — Product: {src['product_category']} | Complaint ID: {src['complaint_id']}\n"
            sources_text += f"{textwrap.shorten(src['text_preview'], width=500, placeholder='...')}\n"

        history.append((None, sources_text))
        yield history

    except Exception as e:
        history[-1] = (message, f"Error: {str(e)}")
        yield history

# Gradio Interface
with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown(
        """
        # CrediTrust Complaint Assistant

        Ask questions about customer complaints across **Credit Cards**, **Personal Loans**, **Savings Accounts**, and **Money Transfers**.
        Answers are generated locally using AI from real customer narratives.
        """
    )

    chatbot = gr.Chatbot(height=650, show_label=False)

    with gr.Row():
        msg = gr.Textbox(
            label="Your Question",
            placeholder="e.g., Why are customers unhappy with Credit Cards?",
            scale=6
        )
        submit = gr.Button("Ask", variant="primary")

    with gr.Row():
        clear = gr.Button("Clear Conversation", variant="secondary")

    gr.Markdown(
        """
        **Important Note**: 
        The **first question** may take **5–15 minutes** to load the full dataset and AI model (Llama 3.2 via Ollama). 
        After that, responses are fast (~10–30 seconds).
        """
    )

    # Events
    submit.click(chat, inputs=[msg, chatbot], outputs=chatbot, queue=True)
    msg.submit(chat, inputs=[msg, chatbot], outputs=chatbot, queue=True)
    clear.click(lambda: [], None, chatbot, queue=False)

# Launch
if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7867,  # Change if port is busy (e.g., 7862)
        share=True,
        inbrowser=True
    )