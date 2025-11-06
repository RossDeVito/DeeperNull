import os
import re
import numpy as np
import pandas as pd

# ===== CONFIG =====
SCORES_DIR = "scores"   # folder containing the four CSVs
CI = 95                 # confidence level

# Known metrics; only the ones present will be used.
REG_METRICS_KNOWN = ["mse", "r2", "mae", "mape", "pearson_r", "spearman_r"]
BIN_METRICS_KNOWN = [
    "average_precision", "roc_auc",
]
# ==================

# ---------- Helpers ----------
def _to_array(x):
    """Parse '[0.1, 0.2, ...]' strings (or lists) into float numpy arrays."""
    if isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=float)
    if pd.isna(x):
        return np.array([], dtype=float)
    s = str(x).strip().strip("[]")
    if not s:
        return np.array([], dtype=float)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            continue
    return np.array(vals, dtype=float)

def _ci_bounds(arr: np.ndarray, ci: int):
    """Return (lower, upper) percentiles for array at CI level; nan if empty."""
    if arr is None or getattr(arr, "size", 0) == 0:
        return (np.nan, np.nan)
    alpha = 100 - ci
    lo_q = alpha / 2
    hi_q = 100 - lo_q
    return (np.percentile(arr, lo_q), np.percentile(arr, hi_q))

# ---------- Parsing ----------
def extract_model_type(desc: str) -> str:
    """Extract model type (e.g., BASIL, PRScs, XGB, etc.) from description."""
    if not isinstance(desc, str):
        return None
    m = re.match(r'([A-Za-z0-9_+-]+)', desc)
    return m.group(1) if m else None

def extract_covar_set(desc: str):
    """
    Parse model_desc strings like:
      'PRScs (lin: age, sex, PCs)' or
      'PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)'
    Returns covariate set (without 'PCs'), or the special unpaired label if null has PCs.
    """
    if not isinstance(desc, str):
        return None
    m = re.search(r'lin:\s*(.+?)(?:,\s*PCs|\))', desc)
    base = m.group(1).strip() if m else None
    if "XGB null:" in desc:
        null_clause = desc.split("XGB null:")[1]
        if re.search(r"\bPCs\b", null_clause):
            return "age, sex, locations, times, PCs for null"
    return base

def has_xgb_null(desc: str) -> bool:
    """Return True if description includes an XGB null model."""
    return isinstance(desc, str) and ("XGB null" in desc)

# ---------- Core Processing ----------
def _process_pair(scores_path: str, boots_path: str, task_label: str,
                  known_metrics: list[str]) -> pd.DataFrame:
    """Load (scores, bootstrap) pair, compute CIs for present metrics, return long DF."""
    scores = pd.read_csv(scores_path)
    boots  = pd.read_csv(boots_path)

    present_metrics = [m for m in known_metrics if (m in boots.columns) and (m in scores.columns)]
    if not present_metrics:
        return pd.DataFrame(columns=[
            "model_desc", "pheno", "model_type", "covar_set", "uses_null",
            "task", "metric", "value", "lower", "upper"
        ])

    # Parse bootstrap arrays and compute CIs
    for m in present_metrics:
        boots[m] = boots[m].apply(_to_array)
        lo_up = boots[m].apply(lambda arr: _ci_bounds(arr, CI))
        boots[f"{m}_lower"] = lo_up.apply(lambda t: t[0])
        boots[f"{m}_upper"] = lo_up.apply(lambda t: t[1])

    # Merge CI columns back
    key_cols = ["model_desc", "pheno"]
    ci_cols = key_cols + [f"{m}_lower" for m in present_metrics] + [f"{m}_upper" for m in present_metrics]
    ci_cols = [c for c in ci_cols if c in boots.columns]
    merged = scores.merge(boots[ci_cols], on=key_cols, how="left")

    # Parse model details
    merged["model_type"] = merged["model_desc"].apply(extract_model_type)
    merged["covar_set"]  = merged["model_desc"].apply(extract_covar_set)
    merged["uses_null"]  = merged["model_desc"].apply(has_xgb_null)

    # Build long format
    rows = []
    for _, row in merged.iterrows():
        for m in present_metrics:
            rows.append({
                "model_desc": row["model_desc"],
                "pheno": row["pheno"],
                "model_type": row.get("model_type"),
                "covar_set": row.get("covar_set"),
                "uses_null": row.get("uses_null"),
                "task": task_label,
                "metric": m,
                "value": row.get(m, np.nan),
                "lower": row.get(f"{m}_lower", np.nan),
                "upper": row.get(f"{m}_upper", np.nan),
            })
    return pd.DataFrame(rows)

# ---------- Wrapper ----------
def build_long_results(scores_dir: str = SCORES_DIR) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with:
      ['model_desc', 'pheno', 'model_type', 'covar_set', 'uses_null',
       'task', 'metric', 'value', 'lower', 'upper']
    """
    reg_scores = os.path.join(scores_dir, "test_scores.csv")
    reg_boots  = os.path.join(scores_dir, "test_boot_scores.csv")
    bin_scores = os.path.join(scores_dir, "test_scores_bin_cls.csv")
    bin_boots  = os.path.join(scores_dir, "test_boot_scores_bin_cls.csv")

    dfs = []
    if os.path.exists(reg_scores) and os.path.exists(reg_boots):
        dfs.append(_process_pair(reg_scores, reg_boots, "regression", REG_METRICS_KNOWN))
    if os.path.exists(bin_scores) and os.path.exists(bin_boots):
        dfs.append(_process_pair(bin_scores, bin_boots, "bin_cls", BIN_METRICS_KNOWN))

    if not dfs:
        return pd.DataFrame(columns=[
            "pheno", "model_type", "covar_set", "uses_null",
            "task", "metric", "value", "lower", "upper"
        ])

    df = pd.concat(dfs, ignore_index=True)
    return df[
        ["pheno", "model_type", "covar_set", "uses_null",
         "task", "metric", "value", "lower", "upper"]
    ]

# ---------- Run ----------
if __name__ == "__main__":
    results_long = build_long_results(SCORES_DIR)
    print(f"{len(results_long):,} rows total\n")
    print(results_long.head(12).to_string(index=False))

    out_path = os.path.join('sup_table', "sup_table_results_long.csv")
    results_long.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
