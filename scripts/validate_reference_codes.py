"""Validate dataset categorical codes against reference_codes.xlsx.

Writes audit artifacts:
- reference_codes__summary.csv
- reference_codes__unknown_codes.csv
- reference_codes__invalid_applies_to.csv
- reference_codes__duplicates.csv
- reference_codes__applies_to_coverage.csv   (NEW: proof applies_to checks ran)
"""
# scripts/validate_reference_codes.py
from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Any

import pandas as pd


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _resolve_from_root(rel: str | Path) -> Path:
    """Resolve a repo-relative path no matter where script is executed from."""
    rel = Path(rel)
    here = Path(__file__).resolve()
    root = here.parents[1]  # repo_root/scripts/this_file.py
    return (root / rel).resolve()


def _ensure_out_dir(out_dir: Path) -> None:
    """Ensure output directory exists."""
    out_dir.mkdir(parents=True, exist_ok=True)


def _coerce_result_to_artifacts(result: Any) -> dict[str, Any]:
    """
    Accept either:
      (A) ReferenceValidationResult-like object with attributes:
          unknown_codes, invalid_applies_to, duplicates, summary
      (B) dict with keys:
          unknown_codes_df, invalid_applies_to_df, duplicates_df, summary_df
          (or unknown_codes/invalid_applies_to/duplicates/summary)

    Returns normalized:
      - unknown_codes_df: pd.DataFrame
      - invalid_applies_to_df: pd.DataFrame
      - duplicates_df: pd.DataFrame
      - summary_df: pd.DataFrame
      - summary_dict: dict (best-effort)
    """
    # Object-style
    if hasattr(result, "unknown_codes") and hasattr(result, "invalid_applies_to") and hasattr(result, "duplicates"):
        unknown_codes_df = getattr(result, "unknown_codes")
        invalid_applies_to_df = getattr(result, "invalid_applies_to")
        duplicates_df = getattr(result, "duplicates")

        summary_obj = getattr(result, "summary", {})
        if isinstance(summary_obj, dict):
            summary_dict = summary_obj
            summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary_obj.items()])
        elif isinstance(summary_obj, pd.DataFrame):
            summary_dict = {}
            summary_df = summary_obj
        else:
            summary_dict = {"summary": str(summary_obj)}
            summary_df = pd.DataFrame([{"metric": "summary", "value": str(summary_obj)}])

        return {
            "unknown_codes_df": unknown_codes_df,
            "invalid_applies_to_df": invalid_applies_to_df,
            "duplicates_df": duplicates_df,
            "summary_df": summary_df,
            "summary_dict": summary_dict,
        }

    # Dict-style
    if isinstance(result, dict):
        unknown_codes_df = result.get("unknown_codes_df", result.get("unknown_codes"))
        invalid_applies_to_df = result.get("invalid_applies_to_df", result.get("invalid_applies_to"))
        duplicates_df = result.get("duplicates_df", result.get("duplicates"))

        summary_df = result.get("summary_df")
        summary_obj = result.get("summary", {})

        summary_dict: dict[str, Any] = {}
        if isinstance(summary_obj, dict):
            summary_dict = summary_obj

        if summary_df is None:
            if isinstance(summary_obj, dict):
                summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary_obj.items()])
            elif isinstance(summary_obj, pd.DataFrame):
                summary_df = summary_obj
            else:
                summary_df = pd.DataFrame([{"metric": "summary", "value": str(summary_obj)}])

        if unknown_codes_df is None or invalid_applies_to_df is None or duplicates_df is None:
            raise ValueError(
                "Validation result dict missing required keys. "
                "Expected unknown_codes(_df), invalid_applies_to(_df), duplicates(_df)."
            )

        return {
            "unknown_codes_df": unknown_codes_df,
            "invalid_applies_to_df": invalid_applies_to_df,
            "duplicates_df": duplicates_df,
            "summary_df": summary_df,
            "summary_dict": summary_dict,
        }

    raise ValueError("Unrecognized validation result type (expected dict or ReferenceValidationResult-like object).")


def _parse_applies_to(raw: str) -> set[str]:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "all":
        return {"all"}
    parts = [p.strip().lower() for p in raw.replace(",", "/").split("/") if p.strip()]
    return set(parts) if parts else {"all"}


def _build_applies_to_coverage(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fields: list[str],
) -> pd.DataFrame:
    """
    Build a per-field coverage report proving applies_to checks were actually possible/executed.

    Columns:
      field, n_rows, n_present, n_known_in_reference, n_checked_context,
      n_invalid_context, allowed_contexts_distinct
    """
    if "record_type" not in df.columns:
        raise ValueError("df missing record_type, cannot compute applies_to coverage")

    ref = ref_df.copy()
    ref.columns = [str(c).strip().lower() for c in ref.columns]
    if "field" not in ref.columns or "code" not in ref.columns:
        raise ValueError("ref_df must include columns: field, code")
    if "applies_to" not in ref.columns:
        # If there is no applies_to column, coverage is not meaningful
        return pd.DataFrame(
            [
                {
                    "field": "__all__",
                    "n_rows": int(len(df)),
                    "n_present": None,
                    "n_known_in_reference": None,
                    "n_checked_context": 0,
                    "n_invalid_context": 0,
                    "allowed_contexts_distinct": 0,
                    "note": "reference has no applies_to column",
                }
            ]
        )

    ref["field"] = ref["field"].astype(str).str.strip().str.lower()
    ref["code"] = ref["code"].astype(str).str.strip()
    ref["applies_to"] = ref["applies_to"].fillna("All").astype(str).str.strip()

    df_rt = df["record_type"].fillna("").astype(str).str.strip().str.lower()

    rows: list[dict[str, Any]] = []
    n_rows_total = int(len(df))

    for field in fields:
        if field not in df.columns:
            continue

        ref_f = ref.loc[ref["field"].eq(field.lower())].copy()
        if ref_f.empty:
            continue

        df_codes = df[field].fillna("").astype(str).str.strip()
        present_mask = df_codes.ne("")
        n_present = int(present_mask.sum())

        # known-in-reference
        ref_codes_l = set(ref_f["code"].astype(str).str.strip().str.lower().tolist())
        df_codes_l = df_codes.str.lower()
        is_known = present_mask & df_codes_l.isin(ref_codes_l)
        n_known = int(is_known.sum())

        # context-checkable only for non-record_type fields
        n_checked = 0
        n_invalid = 0

        allowed_contexts_distinct = int(ref_f["applies_to"].nunique(dropna=True))

        if field != "record_type":
            applies_map = (
                ref_f.assign(code_l=lambda x: x["code"].astype(str).str.strip().str.lower())
                .assign(applies_set=lambda x: x["applies_to"].astype(str).map(_parse_applies_to))
                .set_index("code_l")["applies_set"]
                .to_dict()
            )

            # count checked + invalid
            # (checked means: present and known; invalid means: allowed set doesn't include record_type)
            idxs = df.index[is_known]
            n_checked = int(len(idxs))
            for idx in idxs:
                code_l = str(df.loc[idx, field]).strip().lower()
                allowed = applies_map.get(code_l, {"all"})
                rt = str(df.loc[idx, "record_type"]).strip().lower()
                if "all" not in allowed and rt not in allowed:
                    n_invalid += 1

        rows.append(
            {
                "field": field,
                "n_rows": n_rows_total,
                "n_present": n_present,
                "n_known_in_reference": n_known,
                "n_checked_context": int(n_checked),
                "n_invalid_context": int(n_invalid),
                "allowed_contexts_distinct": allowed_contexts_distinct,
            }
        )

    cov = pd.DataFrame(rows)
    if not cov.empty:
        cov = cov.sort_values(["field"]).reset_index(drop=True)
    return cov


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate dataset categorical codes against reference_codes.xlsx")
    p.add_argument("--data", required=True, help="Path to enriched unified CSV")
    p.add_argument("--reference", required=True, help="Path to reference_codes.xlsx (or .csv export)")
    p.add_argument("--out-dir", required=True, help="Directory to write audit CSV outputs")
    p.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Exit with code 2 if unknown/invalid applies_to codes are found",
    )
    args = p.parse_args(argv)

    data_path = _resolve_from_root(args.data)
    ref_path = _resolve_from_root(args.reference)
    out_dir = _resolve_from_root(args.out_dir)
    _ensure_out_dir(out_dir)

    # ---- imports from your project ----
    try:
        import src.fi.reference_codes as _refcodes  # type: ignore
    except Exception as e:
        print("ERROR: Could not import src.fi.reference_codes. Adjust the import in scripts/validate_reference_codes.py")
        print(f"Import error: {e}")
        return 1

    if not hasattr(_refcodes, "load_reference_codes"):
        print("ERROR: src.fi.reference_codes.load_reference_codes not found.")
        return 1

    load_reference_codes = getattr(_refcodes, "load_reference_codes")
    validate_reference = getattr(_refcodes, "validate_reference_codes", None)

    if not callable(validate_reference):
        print(
            "ERROR: src.fi.reference_codes.validate_reference_codes not found.\n"
            "This script is now standardized on validate_reference_codes(df, ref)."
        )
        return 1

    df = pd.read_csv(data_path)
    ref_df = load_reference_codes(str(ref_path))

    # Run validation
    result = validate_reference(df=df, ref=ref_df)
    artifacts = _coerce_result_to_artifacts(result)

    unknown_codes_df: pd.DataFrame = artifacts["unknown_codes_df"]
    invalid_applies_to_df: pd.DataFrame = artifacts["invalid_applies_to_df"]
    duplicates_df: pd.DataFrame = artifacts["duplicates_df"]
    summary_df: pd.DataFrame = artifacts["summary_df"]
    summary_dict: dict[str, Any] = artifacts["summary_dict"]

    # Build applies_to coverage (proof-of-work)
    # Use validated_fields from summary if available; else default to common categorical columns.
    validated_fields_raw = str(summary_dict.get("validated_fields", "") or "")
    if validated_fields_raw.strip():
        fields = [f.strip() for f in validated_fields_raw.split(",") if f.strip()]
    else:
        fields = ["record_type", "pillar", "category", "value_type", "source_type", "confidence", "gender", "location"]

    coverage_df = _build_applies_to_coverage(df=df, ref_df=ref_df, fields=fields)

    # Write outputs
    unknown_path = out_dir / "reference_codes__unknown_codes.csv"
    invalid_path = out_dir / "reference_codes__invalid_applies_to.csv"
    dupes_path = out_dir / "reference_codes__duplicates.csv"
    summary_path = out_dir / "reference_codes__summary.csv"
    coverage_path = out_dir / "reference_codes__applies_to_coverage.csv"

    unknown_codes_df.to_csv(unknown_path, index=False)
    invalid_applies_to_df.to_csv(invalid_path, index=False)
    duplicates_df.to_csv(dupes_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)

    n_unknown = int(len(unknown_codes_df))
    n_invalid = int(len(invalid_applies_to_df))
    n_dupes = int(len(duplicates_df))

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {unknown_path} (rows={n_unknown})")
    print(f"Wrote: {invalid_path} (rows={n_invalid})")
    print(f"Wrote: {dupes_path} (rows={n_dupes})")
    print(f"Wrote: {coverage_path} (rows={int(len(coverage_df))})")

    if args.fail_on_unknown and (n_unknown > 0 or n_invalid > 0):
        print("FAIL: unknown/invalid applies_to codes found (strict mode).")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
