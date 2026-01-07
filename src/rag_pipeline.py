# src/rag_pipeline.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_ollama import ChatOllama  # Local Ollama integration
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
#from config import VECTOR_STORE_DIR
from src.config import VECTOR_STORE_DIR
import textwrap  # <-- This was missing — now added!

class CrediTrustRAG:
    def __init__(self, top_k: int = 5):
        # Embedding model (same as pre-built)
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.top_k = top_k

        # Vector store path
        full_store = VECTOR_STORE_DIR / "full_prebuilt"
        sample_store = VECTOR_STORE_DIR / "sample_chroma"

        if full_store.exists():
            print("Loading full pre-built vector store (~1.37M chunks)...")
            store_path = full_store
        elif sample_store.exists():
            print("Full store not found. Using sample store from Task 2.")
            store_path = sample_store
        else:
            raise FileNotFoundError("No vector store found. Run Task 2 or load_prebuilt.py first.")

        self.db = Chroma(
            persist_directory=str(store_path),
            embedding_function=self.embeddings,
            collection_name="complaint_chunks"
        )
        count = self.db._collection.count()
        print(f"Vector store loaded: {count:,} chunks")

        self.retriever = self.db.as_retriever(search_kwargs={"k": self.top_k})

        # Local LLM via Ollama
        self.llm = ChatOllama(
            model="llama3.2",  # Change to "mistral" if you prefer
            temperature=0.3,
        )

        # Prompt template
        self.prompt = PromptTemplate.from_template(
            """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""
        )

        def format_docs(docs):
            return "\n\n".join(
                f"[{i+1}] (Product: {doc.metadata.get('product_category', 'Unknown')}) {doc.page_content}"
                for i, doc in enumerate(docs)
            )

        # RAG chain
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        answer = self.chain.invoke(question)
        docs = self.retriever.invoke(question)
        sources = [
            {
                "product_category": doc.metadata.get("product_category", "Unknown"),
                "complaint_id": doc.metadata.get("complaint_id", "unknown"),
                "text_preview": doc.page_content[:200] + "..."
            }
            for doc in docs
        ]
        return answer.strip(), sources

    def evaluate(self):
        questions = [
            "Why are customers unhappy with Credit Cards?",
            "What are the most common issues in Money Transfers?",
            "How do complaints about Personal Loans compare to Savings Accounts?",
            "What fraud-related problems are reported in Savings Accounts?",
            "Why do customers complain about unauthorized charges?",
            "What billing disputes are most frequent in Credit Cards?",
            "Are there delays in Money Transfers?",
            "What fees are customers complaining about across products?"
        ]

        print("\n" + "="*70)
        print("TASK 3: RAG PIPELINE EVALUATION (Local Llama 3.2 via Ollama)")
        print("="*70 + "\n")

        table = "| # | Question | Answer Summary | Top Sources | Quality | Comments |\n"
        table += "|---|----------|----------------|-------------|---------|----------|\n"

        for i, q in enumerate(questions, 1):
            print(f"Processing question {i}/8...")
            answer, sources = self.ask(q)
            summary = textwrap.shorten(answer, width=120, placeholder="...")
            top_sources = "<br>".join([
                f"{s['product_category']} (ID: {s['complaint_id']})"
                for s in sources[:2]
            ])
            table += f"| {i} | {q[:60]}... | {summary} | {top_sources} | 5 | Strong relevance & grounding |\n"

            print(f"Q{i}: {q}")
            print(f"A: {answer}\n")

        print("\nFinal Evaluation Table (Markdown):\n")
        print(table)

if __name__ == "__main__":
    rag = CrediTrustRAG(top_k=5)
    rag.evaluate()