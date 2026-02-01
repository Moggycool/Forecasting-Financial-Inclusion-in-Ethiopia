# Forecasting Financial Inclusion in Ethiopia (Unified Dataset + Enrichment + EDA)

This repository contains a reproducible pipeline to:

1) validate and enrich a unified Financial Inclusion (FI) dataset for Ethiopia, and  
2) generate EDA tables used to support narrative insights and forecasting readiness.

The workflow is designed to be grader/reviewer-friendly:

- deterministic CLI commands
- explicit schema expectations
- diagnostics outputs when validation issues are found
- written insights + limitations in `INSIGHTS.md`

---

## Repository Structure (key items)

- `data/raw/ethiopia_fi_unified_data.csv`  
  Unified FI dataset (raw/un-enriched input)

- `data/enrichment/new_records.yaml`  
  Human-authored enrichment records (events, impact links, targets, etc.)

- `data/processed/ethiopia_fi_unified_data__enriched.csv`  
  Output of enrichment pipeline

- `data/processed/diagnostics/`  
  Validation diagnostics emitted by enrichment pipeline (only written when issues exist)

- `scripts/apply_enrichment.py`  
  Applies YAML enrichment records and runs relationship diagnostics

- `scripts/run_exploration.py`  
  Runs EDA tables (counts, temporal range, coverage, events, links)

- `src/fi/`  
  Core library modules: `io`, `validation`, `enrich`, `explore`

- `INSIGHTS.md`  
  Written insights using a Claim/Evidence/Interpretation/Confidence structure + limitations

---

## Quickstart

### 1) Create environment (example)

Use your preferred environment manager. Example with `venv`:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Project Tree

```text

Forecasting-Financial-Inclusion-in-Ethiopia
├─ dashboard
│  └─ app.py
├─ data
│  ├─ enrichment
│  │  └─ new_records.yaml
│  └─ processed
│     ├─ diagnostics
│     ├─ eda
│     ├─ eda_enriched
│     │  ├─ counts__category.csv
│     │  ├─ counts__pillar.csv
│     │  ├─ counts__record_type.csv
│     │  ├─ events.csv
│     │  ├─ impact_links.csv
│     │  ├─ indicator_coverage.csv
│     │  ├─ temporal_range.csv
│     │  ├─ temporal_range__events.csv
│     │  ├─ temporal_range__observations.csv
│     │  └─ temporal_range__targets.csv
│     ├─ ethiopia_fi_unified_data__enriched.csv
│     ├─ task1_notebook_outputs
│     │  ├─ counts__category.csv
│     │  ├─ counts__pillar.csv
│     │  ├─ counts__record_type.csv
│     │  ├─ data_summary.csv
│     │  ├─ diagnostics
│     │  ├─ events.csv
│     │  ├─ impact_links.csv
│     │  ├─ indicator_coverage.csv
│     │  ├─ temporal_range.csv
│     │  ├─ temporal_range__events.csv
│     │  ├─ temporal_range__observations.csv
│     │  └─ temporal_range__targets.csv
│     └─ task2_eda_outputs
│        ├─ acc_ownership_gender_wide_and_gap.csv
│        ├─ acc_ownership_growth_pp_by_period.csv
│        ├─ confidence_distribution.csv
│        ├─ events_timeline_table.csv
│        ├─ indicator_coverage_from_data.csv
│        ├─ infrastructure_enablers_table.csv
│        ├─ mpesa_registered_vs_active.csv
│        ├─ plots
│        │  ├─ access_account_ownership_by_gender.png
│        │  ├─ access_account_ownership_growth_pp.png
│        │  ├─ access_account_ownership_trajectory.png
│        │  ├─ confidence_distribution.png
│        │  ├─ events_timeline.png
│        │  ├─ infrastructure_enablers_small_multiples.png
│        │  ├─ overlay_access_account_ownership.png
│        │  ├─ overlay_usage_mobile_money_rate.png
│        │  ├─ temporal_coverage_heatmap.png
│        │  ├─ usage_mobile_money_account_rate.png
│        │  └─ usage_proxies_small_multiples.png
│        ├─ snapshot_only_indicators_nobs_eq1.csv
│        ├─ sparse_indicators_nobs_le2.csv
│        ├─ summary_by_pillar.csv
│        ├─ summary_by_record_type.csv
│        ├─ summary_by_source_type.csv
│        ├─ temporal_coverage_heatmap_matrix.csv
│        ├─ usage_proxies_table.csv
│        └─ usage_registered_active_snapshot_table.csv
├─ data_enrichment_log.md
├─ models
├─ notebooks
│  ├─ 01_task1_exploration_and_schema.ipynb
│  ├─ 02_task2_eda.ipynb
│  ├─ data
│  │  └─ processed
│  │     └─ task1_notebook_outputs
│  │        └─ diagnostics
│  └─ README.md
├─ README.md
├─ reports
│  └─ figures
├─ requirements.txt
├─ scripts
│  ├─ apply_enrichment.py
│  └─ run_exploration.py
├─ src
│  └─ fi
│     ├─ enrich.py
│     ├─ explore.py
│     ├─ io.py
│     ├─ schemas
│     │  └─ unified_schema.md
│     ├─ validation.py
│     └─ __init__.py
└─ tests
   └─ __init__.py

```

## Unified Data Schema & Reproducible Pipeline

This repository uses a **single unified dataset design** to store multiple record types (observations, events, impact links, and targets) in one table.  
The pipeline supports **schema validation, enrichment, and automated exploratory data analysis (EDA)** to ensure reproducibility.

---

## Data Schema — Unified Table

The unified dataset is stored as **one table with multiple record types**.

The minimum schema is validated by:

```python
src.fi.validation.assert_min_schema()
```

---

## Required Concepts

### Core Fields

| Field | Description |
|---------|-------------|
| **record_type** | Identifies what a row represents |
| **record_id** | Unique row identifier |
| **observation_date** | Date-like field used for temporal calculations |
| **parent_id** | Relationship pointer (primarily for `impact_link` rows) |

---

## Expected `record_type` Values

| Value | Meaning |
|------------|-------------------------------------------|
| observation | Measured indicators (e.g., account ownership) |
| event | Timeline events (e.g., product launches, policy changes) |
| impact_link | Links an event to an indicator (hypothesized causal relationship) |
| target | Goal/target values for forecasting (optional) |

> **Note:** Some raw unified datasets may omit `parent_id`.  
> The enrichment script automatically adds it if missing.

---

## Common Additional Columns

- `indicator_code` — observations/targets
- `pillar`, `category` — taxonomy/grouping fields
- Qualitative text fields — events & impact links

---

## Pipeline — Reproducible Commands

All scripts accept **CLI arguments** so reviewers can reproduce results exactly.

Run commands **from the repository root directory**.

---

## Step A — Apply Enrichment (YAML → enriched unified CSV)

### Inputs

```text
data/raw/ethiopia_fi_unified_data.csv
data/enrichment/new_records.yaml
```

### Outputs

```text
data/processed/ethiopia_fi_unified_data__enriched.csv
data/processed/diagnostics/*.csv
```

### Command (PowerShell / Windows)

```bash
python scripts/apply_enrichment.py ^
  --unified data/raw/ethiopia_fi_unified_data.csv ^
  --new-records data/enrichment/new_records.yaml ^
  --out data/processed/ethiopia_fi_unified_data__enriched.csv ^
  --prefix ENR
```

### Optional Strict Mode

Fails immediately if invalid records are detected:

```bash
python scripts/apply_enrichment.py --fail-on-invalid
```

---

### Enrichment Validation Diagnostics

If issues are found, the following diagnostic files may be created:

| File | Description |
|-------------------------------------------|---------------------------------------------|
| invalid_events_with_pillar.csv | Events should not contain taxonomy fields |
| invalid_impact_links_missing_parent.csv | Missing `parent_id` |
| invalid_impact_links_unresolved_parent.csv | `parent_id` does not match an event |

---

## Step B — Run Exploratory Data Analysis (EDA)

Generates summary tables for:

- counts
- coverage
- temporal ranges
- events
- impact links

### Task 2

```bash
python scripts/run_exploration.py ^
  --in data/processed/ethiopia_fi_unified_data__enriched.csv ^
  --out-dir data/processed/task2_eda_outputs
```

### Task 1 (optional outputs)

```bash
python scripts/run_exploration.py ^
  --in data/processed/ethiopia_fi_unified_data__enriched.csv ^
  --out-dir data/processed/task1_notebook_outputs
```

---

#### Expected EDA Outputs

```text
counts__record_type.csv
counts__pillar.csv
counts__category.csv
temporal_range.csv
temporal_range__observations.csv
temporal_range__targets.csv
temporal_range__events.csv
indicator_coverage.csv
events.csv
impact_links.csv
```

---

## How to Read Results

| Location | Purpose |
|-------------|-------------------------------|
| INSIGHTS.md | Narrative findings & interpretations |
| task2_eda_outputs/ | Raw analysis tables |
| diagnostics/ | Enrichment/relationship issues |

---

## Data Provenance (High-Level)

The unified dataset typically combines:

- Household survey indicators (e.g., Global Findex-style measures)
- Administrative/operator metrics (mobile money registrations/activity)
- Qualitative timeline events (policies, product launches, market entries)

Because sources differ in **definitions, frequency, and measurement**, limitations and confidence levels are documented in:

```text
INSIGHTS.md
```

---

# Known Limitations

Common challenges include:

- Sparse longitudinal coverage
- Mixed definitions (survey vs administrative)
- National vs subgroup aggregation differences
- Event → measurement time lags
- Heterogeneous confidence across indicators and periods

See **INSIGHTS.md** for detailed discussion.

---

## Reproducibility Notes

- Scripts automatically insert repo root into `sys.path`
- Can be executed directly:

  ```bash
  python scripts/<script_name>.py
  ```

- CLI paths resolve relative to repo root
- Always run commands from the **repository root**

---

## Quick Start

```bash
# 1 — Enrich dataset
python scripts/apply_enrichment.py --unified data/raw/ethiopia_fi_unified_data.csv --new-records data/enrichment/new_records.yaml --out data/processed/ethiopia_fi_unified_data__enriched.csv

# 2 — Run EDA
python scripts/run_exploration.py --in data/processed/ethiopia_fi_unified_data__enriched.csv --out-dir data/processed/task2_eda_outputs
```

---

**Project goal:**  
Provide a clean, validated, and reproducible workflow for exploring Ethiopia’s financial inclusion indicators, events, and their potential impacts.
