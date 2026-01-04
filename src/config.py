from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Files
ZIP_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
ZIP_PATH = RAW_DIR / "complaints.csv.zip"
RAW_CSV = RAW_DIR / "complaints.csv"

FILTERED_CSV = PROCESSED_DIR / "filtered_complaints.csv"
WORD_COUNT_PLOT = NOTEBOOKS_DIR / "narrative_word_count_distribution.png"

# Pre-built embeddings provided in the challenge
PREBUILT_PARQUET = DATA_DIR / "complaint_embeddings.parquet"  # ← Add this line

# Vector store paths
SAMPLE_VECTOR_STORE = VECTOR_STORE_DIR / "sample_chroma"
FULL_PREBUILT_STORE = VECTOR_STORE_DIR / "full_prebuilt"

RELEVANT_CFPB_PRODUCTS = [
    "Credit card or prepaid card",
    "Payday loan, title loan, or personal loan",
    "Consumer Loan",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service"
]

PRODUCT_MAPPING = {
    "Credit card or prepaid card": "Credit Cards",
    "Payday loan, title loan, or personal loan": "Personal Loans",
    "Consumer Loan": "Personal Loans",
    "Checking or savings account": "Savings Accounts",
    "Money transfer, virtual currency, or money service": "Money Transfers"
}