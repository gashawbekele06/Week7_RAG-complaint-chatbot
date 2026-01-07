# temp_inspect_parquet.py
import pandas as pd
from config import PREBUILT_PARQUET

df = pd.read_parquet(PREBUILT_PARQUET, columns=[], engine='pyarrow')  # Load only metadata
print("Columns in the parquet file:")
print(df.columns.tolist())

# Show first few rows structure
print("\nFirst row sample:")
print(df.head(1).to_dict())