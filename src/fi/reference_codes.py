""" A module for loading and validating reference codes against dataframes. """
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ReferenceValidationResult:
    """Result of reference code validation."""
    unknown_codes: pd.DataFrame
    invalid_applies_to: pd.DataFrame
    duplicates: pd.DataFrame
    summary: dict


def load_reference_codes(path: str) -> pd.DataFrame:
    """
    Load reference codes from .xlsx or .csv.

    Expected columns (case-insensitive):
      - field
      - code
      - description (optional but recommended)
      - applies_to (optional but recommended)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference codes file not found: {p}")

    if p.suffix.lower() in {".xlsx", ".xls"}:
        ref = pd.read_excel(p)  # reads first sheet by default
    elif p.suffix.lower() == ".csv":
        ref = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported reference codes format: {p.suffix}")

    ref = ref.copy()
    ref.columns = [str(c).strip().lower() for c in ref.columns]

    required = {"field", "code"}
    missing = sorted(list(required - set(ref.columns)))
    if missing:
        raise ValueError(f"reference codes missing required columns: {missing}")

    ref["field"] = ref["field"].astype(str).str.strip()
    ref["code"] = ref["code"].astype(str).str.strip()

    if "applies_to" in ref.columns:
        ref["applies_to"] = ref["applies_to"].fillna("All").astype(str).str.strip()

    return ref


def _norm_str_series(s: pd.Series) -> pd.Series:
    """Normalize a string series by filling NaN with empty string and stripping whitespace."""
    return s.fillna("").astype(str).str.strip()


def _parse_applies_to(raw: str) -> set[str]:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "all":
        return {"all"}
    # support "event/observation" or "event, observation"
    parts = [p.strip().lower() for p in raw.replace(",", "/").split("/") if p.strip()]
    return set(parts) if parts else {"all"}


def validate_reference_codes(
    df: pd.DataFrame,
    ref: pd.DataFrame,
    fields: Iterable[str] = (
        "record_type",
        "pillar",
        "category",
        "value_type",
        "source_type",
        "confidence",
        "gender",
        "location",
    ),
) -> ReferenceValidationResult:
    """
    Validate that codes used in df are present in the reference table and, where possible,
    that they are used in the correct record_type context via applies_to.

    Returns DataFrames of violations + summary dict.
    """
    if "record_type" not in df.columns:
        raise ValueError("df is missing required column: record_type")

    if "field" not in ref.columns or "code" not in ref.columns:
        raise ValueError("ref must contain columns: field, code")

    df_rt = _norm_str_series(df["record_type"]).str.lower()

    ref_norm = ref.copy()
    ref_norm["field"] = ref_norm["field"].astype(str).str.strip().str.lower()
    ref_norm["code"] = ref_norm["code"].astype(str).str.strip()

    # detect duplicates in reference
    dup_mask = ref_norm.duplicated(subset=["field", "code"], keep=False)
    duplicates = ref.loc[dup_mask].sort_values(["field", "code"]).reset_index(drop=True)

    unknown_rows: list[pd.DataFrame] = []
    invalid_rows: list[pd.DataFrame] = []

    for field in fields:
        if field == "record_type":
            continue
        if field not in df.columns:
            continue

        ref_f = ref_norm.loc[ref_norm["field"].eq(field.lower())].copy()
        if ref_f.empty:
            continue

        ref_codes = set(ref_f["code"].astype(str).str.strip().str.lower().tolist())

        df_codes = _norm_str_series(df[field]).str.lower()
        present_mask = df_codes.ne("")  # ignore blanks

        # unknown codes
        is_unknown = present_mask & ~df_codes.isin(ref_codes)
        if is_unknown.any():
            unknown_rows.append(
                pd.DataFrame(
                    {
                        "field": field,
                        "record_type": df_rt.loc[is_unknown].values,
                        "code_used": df.loc[is_unknown, field].astype(str).values,
                    }
                )
            )

        # applies_to checks (if provided)
        if "applies_to" in ref_norm.columns:
            applies_map = (
                ref_norm.loc[ref_norm["field"].eq(field.lower()), ["code", "applies_to"]]
                .assign(code_l=lambda x: x["code"].astype(str).str.strip().str.lower())
                .assign(applies_set=lambda x: x["applies_to"].astype(str).map(_parse_applies_to))
                .set_index("code_l")["applies_set"]
                .to_dict()
            )

            # Only validate rows whose code exists in reference (unknowns handled above)
            is_known = present_mask & df_codes.isin(ref_codes)
            if is_known.any():
                bad_idx = []
                applies_vals = []
                for idx in df.index[is_known]:
                    code_l = str(df.loc[idx, field]).strip().lower()
                    allowed = applies_map.get(code_l, {"all"})
                    rt = str(df.loc[idx, "record_type"]).strip().lower()
                    if "all" not in allowed and rt not in allowed:
                        bad_idx.append(idx)
                        applies_vals.append("/".join(sorted(allowed)))

                if bad_idx:
                    invalid_rows.append(
                        pd.DataFrame(
                            {
                                "field": field,
                                "record_type": df.loc[bad_idx, "record_type"].astype(str).values,
                                "code_used": df.loc[bad_idx, field].astype(str).values,
                                "applies_to": applies_vals,
                            }
                        )
                    )

    unknown_codes = (
        pd.concat(unknown_rows, ignore_index=True)
        if unknown_rows
        else pd.DataFrame(columns=["field", "record_type", "code_used"])
    )

    invalid_applies_to = (
        pd.concat(invalid_rows, ignore_index=True)
        if invalid_rows
        else pd.DataFrame(columns=["field", "record_type", "code_used", "applies_to"])
    )

    summary = {
        "n_rows": int(len(df)),
        "n_unknown_codes": int(len(unknown_codes)),
        "n_invalid_applies_to": int(len(invalid_applies_to)),
        "n_reference_duplicates": int(len(duplicates)),
        "validated_fields": ",".join([f for f in fields if f in df.columns]),
    }

    return ReferenceValidationResult(
        unknown_codes=unknown_codes,
        invalid_applies_to=invalid_applies_to,
        duplicates=duplicates,
        summary=summary,
    )