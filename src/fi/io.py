""" Input/output functions for dataframes in the FI schema."""
from __future__ import annotations

import pandas as pd



def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV from disk."""
    return pd.read_csv(path, dtype=str).assign(
        value_numeric=lambda d: pd.to_numeric(d["value_numeric"], errors="coerce")
    )

def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a CSV to disk."""
    df.to_csv(path, index=False)
