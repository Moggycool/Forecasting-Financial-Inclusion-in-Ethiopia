""" Script to apply enrichment records from YAML to unified FI dataset. """
from __future__ import annotations

# pylint: disable=wrong-import-position
import sys
from pathlib import Path

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import pandas as pd
# pylint: enable=wrong-import-position

from src.fi.enrich import append_records, ensure_record_ids, load_new_records_yaml
from src.fi.io import load_csv, save_csv
from src.fi.validation import (
    assert_min_schema,
    invalid_events_with_pillar,
    invalid_impact_links_missing_parent,
    invalid_impact_links_unresolved_parent,
)

DEFAULT_UNIFIED = Path("data/raw/ethiopia_fi_unified_data.csv")
DEFAULT_NEW_RECORDS = Path("data/enrichment/new_records.yaml")
DEFAULT_OUT = Path("data/processed/ethiopia_fi_unified_data__enriched.csv")


def _ensure_dirs(path: Path) -> None:
    """Ensure parent directories exist for a given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns required by validation exist.
    Some raw/unified datasets may not include relationship columns (e.g. parent_id).
    """
    required = ["parent_id"]
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return df

    df = df.copy()
    for c in missing:
        df[c] = pd.NA
    return df

def _resolve_from_root(p: Path) -> Path:
    """Resolve relative paths from the repository root."""
    return p if p.is_absolute() else (_ROOT / p)

def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Apply YAML enrichment records to unified FI dataset."
    )
    ap.add_argument(
        "--unified",
        type=str,
        default=str(DEFAULT_UNIFIED),
        help="Path to unified CSV",
    )
    ap.add_argument(
        "--new-records",
        type=str,
        default=str(DEFAULT_NEW_RECORDS),
        help="Path to YAML with new records",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="Output path for enriched unified CSV",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="ENR",
        help="Prefix for auto-generated record_id values",
    )
    ap.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="If set, abort when validation finds any invalid records.",
    )
    args = ap.parse_args()

    # unified_path = Path(args.unified)
    unified_path = _resolve_from_root(Path(args.unified))
    new_records_path = _resolve_from_root(Path(args.new_records))
    out_path = _resolve_from_root(Path(args.out))
    if not unified_path.exists():
        raise SystemExit(f"Unified input file does not exist: {unified_path}")
    if not new_records_path.exists():
        raise SystemExit(f"New records YAML file does not exist: {new_records_path}")   
    

    print(f"[load] unified: {unified_path}")
    unified = load_csv(str(unified_path))
    unified = _ensure_required_columns(unified)
    assert_min_schema(unified)

    print(f"[load] new records YAML: {new_records_path}")
    new_df = load_new_records_yaml(str(new_records_path))
    new_df = ensure_record_ids(new_df, prefix=args.prefix)
    new_df = _ensure_required_columns(new_df)

    # Align + append
    enriched = append_records(unified=unified, new_records=new_df)

    # Re-check schema presence after append
    enriched = _ensure_required_columns(enriched)
    assert_min_schema(enriched)

    # Validation diagnostics
    bad_events = invalid_events_with_pillar(enriched)
    bad_links_missing_parent = invalid_impact_links_missing_parent(enriched)
    bad_links_unresolved = invalid_impact_links_unresolved_parent(enriched)

    print("[validate] invalid events with non-empty pillar:", len(bad_events))
    print("[validate] impact_links missing parent_id:", len(bad_links_missing_parent))
    print("[validate] impact_links unresolved parent_id:", len(bad_links_unresolved))

    # Save diagnostics
    diag_dir = out_path.parent / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if len(bad_events) > 0:
        save_csv(bad_events, str(diag_dir / "invalid_events_with_pillar.csv"))
    if len(bad_links_missing_parent) > 0:
        save_csv(
            bad_links_missing_parent,
            str(diag_dir / "invalid_impact_links_missing_parent.csv"),
        )
    if len(bad_links_unresolved) > 0:
        save_csv(
            bad_links_unresolved,
            str(diag_dir / "invalid_impact_links_unresolved_parent.csv"),
        )

    if args.fail_on_invalid and (
        len(bad_events) + len(bad_links_missing_parent) + len(bad_links_unresolved) > 0
    ):
        raise SystemExit(
            "Validation failed (see diagnostics files). Re-run without --fail-on-invalid to still write output."
        )

    _ensure_dirs(out_path)
    print(f"[write] enriched unified: {out_path}")
    save_csv(enriched, str(out_path))

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())