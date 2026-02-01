""" Tests for reference code validation. """
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))



from src.fi.reference_codes import validate_reference_codes


def test_unknown_code_detected():
    """ Test that unknown reference codes are detected. """
    ref = pd.DataFrame({
        "field": ["pillar"],
        "code": ["ACCESS"],
        "description": ["ok"],
        "applies_to": ["observation/target/impact_link"],
    })

    df = pd.DataFrame({
        "record_type": ["observation"],
        "pillar": ["ACCESSX"],  # unknown
        "record_id": ["R1"],
        "observation_date": ["2020-01-01"],
    })

    res = validate_reference_codes(df, ref, fields=["pillar"])
    assert len(res.unknown_codes) == 1


def test_applies_to_enforced():
    """ Test that applies_to restrictions are enforced. """
    ref = pd.DataFrame({
        "field": ["category"],
        "code": ["policy"],
        "description": ["event only"],
        "applies_to": ["event"],
    })

    df = pd.DataFrame({
        "record_type": ["observation"],  # invalid for category=policy
        "category": ["policy"],
        "record_id": ["R1"],
        "observation_date": ["2020-01-01"],
    })

    res = validate_reference_codes(df, ref, fields=["category"])
    assert len(res.invalid_applies_to) == 1


def test_reference_duplicates_detected():
    """ Test that duplicate reference codes are detected. """
    ref = pd.DataFrame({
        "field": ["pillar", "pillar"],
        "code": ["ACCESS", "ACCESS"],
        "description": ["a", "b"],
        "applies_to": ["All", "All"],
    })

    df = pd.DataFrame({
        "record_type": ["observation"],
        "pillar": ["ACCESS"],
        "record_id": ["R1"],
        "observation_date": ["2020-01-01"],
    })

    res = validate_reference_codes(df, ref, fields=["pillar"])
    assert len(res.duplicates) == 2

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))