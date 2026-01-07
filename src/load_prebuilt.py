# src/load_prebuilt.py
import pandas as pd
import pyarrow.parquet as pq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from config import PREBUILT_PARQUET, VECTOR_STORE_DIR

def load_parquet_to_chroma(batch_size=5000):  # Small enough for your file
    if not PREBUILT_PARQUET.exists():
        print(f"ERROR: File not found: {PREBUILT_PARQUET}")
        return False

    print(f"Loading {PREBUILT_PARQUET.name} with batch_size={batch_size}...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db_path = VECTOR_STORE_DIR / "full_prebuilt"
    db_path.mkdir(parents=True, exist_ok=True)

    # Start fresh
    db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name="complaint_chunks"
    )
    db.delete_collection()  # Clear old data
    db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name="complaint_chunks"
    )

    total = 0
    try:
        parquet_file = pq.ParquetFile(PREBUILT_PARQUET)
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            batch_df = batch.to_pandas()
            print(f"Batch {i+1}: {len(batch_df)} rows — indexing...")

            docs = []
            for _, row in batch_df.iterrows():
                text = row['document']  # Confirmed from your columns
                if pd.isna(text) or not text:
                    continue

                metadata = {
                    "complaint_id": str(row.get('complaint_id', 'unknown')),
                    "product_category": str(row.get('product_category', 'Unknown')),
                    "product": str(row.get('product', 'Unknown')),
                    "issue": str(row.get('issue', '')),
                    "sub_issue": str(row.get('sub_issue', '')),
                    "company": str(row.get('company', '')),
                    "state": str(row.get('state', '')),
                    "date_received": str(row.get('date_received', '')),
                    "chunk_index": int(row.get('chunk_index', 0)),
                    "total_chunks": int(row.get('total_chunks', 1)),
                }

                doc = Document(page_content=str(text), metadata=metadata)
                docs.append(doc)

            db.add_documents(docs)
            total += len(docs)
            print(f"Indexed {total:,} chunks so far")

        print(f"\nSUCCESS! Full vector store created with {total:,} chunks")
        print(f"Location: {db_path}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    load_parquet_to_chroma(batch_size=5000)  # Safe for your file's row groups