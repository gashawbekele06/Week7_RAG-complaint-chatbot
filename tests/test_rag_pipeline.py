# tests/test_rag_pipeline.py
import pytest
from pathlib import Path
from src.rag_pipeline import CrediTrustRAG
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag_system():
    """Create RAG instance once per module (expensive to load every time)"""
    # Use sample store for faster tests
    # If you want full store tests → change to full_prebuilt and increase timeouts
    rag = CrediTrustRAG(top_k=3)
    # Override to sample store for speed in CI/tests
    rag.db = rag.db.__class__(
        persist_directory=str(Path("vector_store") / "sample_chroma"),
        embedding_function=rag.embeddings,
        collection_name="complaint_chunks"
    )
    return rag


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Customer complains about high interest rate on personal loan.",
            metadata={"product_category": "Personal Loan",
                      "complaint_id": "12345"}
        ),
        Document(
            page_content="Unauthorized transaction on credit card last week.",
            metadata={"product_category": "Credit Card",
                      "complaint_id": "67890"}
        )
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rag_initialization(rag_system):
    """Check that RAG initializes without crashing"""
    assert rag_system is not None
    assert rag_system.retriever is not None
    assert rag_system.llm is not None
    assert rag_system.prompt is not None


def test_retriever_returns_docs(rag_system):
    """Basic retriever smoke test"""
    docs = rag_system.retriever.invoke("credit card problems")
    assert isinstance(docs, list)
    assert len(docs) <= rag_system.top_k
    if len(docs) > 0:
        assert hasattr(docs[0], "page_content")
        assert hasattr(docs[0], "metadata")


def test_ask_returns_tuple(rag_system):
    """ask() should return (str, list)"""
    answer, sources = rag_system.ask(
        "Why are customers unhappy with Credit Cards?")
    assert isinstance(answer, str)
    assert len(answer) > 10, "Answer seems too short"
    assert isinstance(sources, list)
    assert all(isinstance(s, dict) for s in sources)
    assert all("product_category" in s for s in sources)


def test_prompt_contains_context_and_question(rag_system):
    """Quick check that prompt template has placeholders"""
    prompt_text = rag_system.prompt.template
    assert "{context}" in prompt_text
    assert "{question}" in prompt_text


def test_format_docs(rag_system, sample_docs):
    """Test document formatting helper"""
    formatted = rag_system.format_docs(sample_docs)
    assert isinstance(formatted, str)
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "Personal Loan" in formatted
    assert "Credit Card" in formatted


@pytest.mark.slow
def test_end_to_end_ask(rag_system):
    """End-to-end test — can be slow because it actually calls LLM"""
    answer, sources = rag_system.ask("billing dispute credit card")
    assert isinstance(answer, str)
    assert len(answer) > 20
    assert any("Credit Card" in s["product_category"] for s in sources)
