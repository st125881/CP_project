"""Shared feature engineering, used by both training and the app."""
import numpy as np
import pandas as pd

FEATURES = [
    "energy_difference",
    "time_temperature_integrated",
    "dc_internal_resistance",
    "charge_energy",
    "temperature_minimum",
    "temperature_range",
]
N_CYCLES = 50


def first_50_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first 50 records per battery (same logic as the notebook)."""
    df = df.copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date_time_iso"] = pd.to_datetime(df["date_time_iso"], errors="coerce")
    df = df.sort_values(["Battery", "date_time_iso"])
    first50 = df.groupby("Battery", group_keys=False).head(N_CYCLES).copy()
    counts = first50.groupby("Battery")["cycle_index"].nunique()
    valid = counts[counts >= N_CYCLES].index
    return first50[first50["Battery"].isin(valid)].copy()


def battery_features(first50: pd.DataFrame) -> pd.DataFrame:
    """One row per battery: mean of each feature over its first 50 cycles."""
    out = first50.groupby("Battery")[FEATURES].mean()
    return out.replace([np.inf, -np.inf], np.nan)


def features_from_cycle_file(cycles: pd.DataFrame) -> pd.DataFrame:
    """Turn a single battery's per-cycle table into a 1-row feature frame."""
    missing = [c for c in FEATURES if c not in cycles.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    cycles = cycles.copy()
    if "cycle_index" in cycles.columns:
        cycles = cycles.sort_values("cycle_index")
    cycles = cycles.head(N_CYCLES)
    if len(cycles) < N_CYCLES:
        raise ValueError(f"Need {N_CYCLES} cycles, file has {len(cycles)}.")
    vals = cycles[FEATURES].apply(pd.to_numeric, errors="coerce").mean()
    return pd.DataFrame([vals.replace([np.inf, -np.inf], np.nan)])
