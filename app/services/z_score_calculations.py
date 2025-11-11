# app/services/z_score_calculations.py

"""
Helpers for converting raw per-game stats into standard fantasy
basketball z-scores.

This is based on your old per-game NBA baselines:
- League means / stds for counting stats (PTS, FG3M, REB, AST, STK, BLK, TOV)
- FG% and FT% treated as "impact" of makes vs. league-average percentage,
  scaled by attempts (FGM/FGA, FTM/FTA), then z-scored.

If any of the relevant inputs for a category are NaN (or "NaN" as a string,
or missing/empty), the z-score for that category will be NaN, and AVG_Z will
be NaN if any component z is NaN.
"""

from __future__ import annotations
from typing import Dict
import math

# -----------------------
# Baselines (from old file)
# -----------------------

LEAGUE_MEANS: Dict[str, float] = {
    "PTS": 16.714,
    "FG3M": 1.876,
    "REB": 6.000,
    "AST": 3.825,
    "STL": 0.970,
    "BLK": 0.680,
    "TOV": 1.883,
    "FGM": 5.711,
    "FGA": 11.943,
    "FTM": 2.497,
    "FTA": 3.126,
}

LEAGUE_STDS: Dict[str, float] = {
    "PTS": 6.078,
    "FG3M": 0.588,
    "REB": 2.500,
    "AST": 2.077,
    "STL": 0.435,
    "BLK": 0.571,
    "TOV": 0.891,
    "FGM": 2.110,
    "FGA": 4.494,
    "FTM": 1.615,
    "FTA": 1.947,
}

# League-average shooting percentages (from means)
LG_FG_PCT: float = (
    LEAGUE_MEANS["FGM"] / LEAGUE_MEANS["FGA"] if LEAGUE_MEANS["FGA"] else 0.0
)
LG_FT_PCT: float = (
    LEAGUE_MEANS["FTM"] / LEAGUE_MEANS["FTA"] if LEAGUE_MEANS["FTA"] else 0.0
)

# Baselines for "impact" of FG% and FT% (from your old code)
FG_IMPACT_MEAN: float = -0.087797348
FG_IMPACT_STD: float = 0.690900598
FT_IMPACT_MEAN: float = 0.029394472
FT_IMPACT_STD: float = 0.303514900


def _safe_float(v) -> float:
    """
    Convert to float, but:

    - If the value is already NaN, keep it as NaN.
    - If the string 'NaN' / 'nan' / 'null' / 'none' or '' is passed, return NaN.
    - If the value is None, return NaN.
    - Otherwise, try float(v); if that fails, return NaN.

    This is intentionally strict so that "missing-ish" values don't silently
    turn into 0.0 and fake a real z-score.
    """
    # Explicit None -> NaN
    if v is None:
        return float("nan")

    # Already a float
    if isinstance(v, float):
        return v if not math.isnan(v) else float("nan")

    # Int or other numeric
    if isinstance(v, int):
        return float(v)

    # Strings and other types
    try:
        s = str(v).strip().lower()
        if s == "" or s in ("nan", "none", "null"):
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _z_scalar(value: float, mean: float, std: float, invert: bool = False) -> float:
    """
    Simple scalar z-score:
      z = (value - mean) / std
    If invert=True, returns -z (useful for TOV where lower is better).

    If value is NaN, returns NaN.
    """
    val = _safe_float(value)

    if math.isnan(val):
        return float("nan")

    m = float(mean)
    s = float(std) if std not in (None, 0) else 0.0
    if s == 0.0:
        # If std is 0, the category has no spread; treat as neutral (0 z)
        return 0.0

    z = (val - m) / s
    return -z if invert else z


def raw_to_z_scores(avg_raw: Dict[str, float]) -> Dict[str, float]:
    """
    Convert per-game raw stats into standard z-scores relative to
    fixed NBA-wide baselines.

    Input:
        avg_raw: per-game stats, for example:
          {
            "PTS": 18.5,
            "FG3M": 2.4,
            "REB": 7.1,
            "AST": 3.5,
            "STL": 1.1,
            "BLK": 0.8,
            "TOV": 2.1,
            "FGM": 7.0,
            "FGA": 14.5,
            "FTM": 3.5,
            "FTA": 4.2,
          }

    Output:
        {
          "Z_PTS": float,
          "Z_FG3M": float,
          "Z_REB": float,
          "Z_AST": float,
          "Z_STL": float,
          "Z_BLK": float,
          "Z_TOV": float,      # already inverted so lower TOV => higher z
          "Z_FG_PCT": float,
          "Z_FT_PCT": float,
          "AVG_Z": float,
        }

    If any input needed for a category is NaN / "NaN" / missing / empty,
    that category's z will be NaN, and AVG_Z will be NaN if any component
    z is NaN.
    """
    # Pull values safely (with NaN propagation)
    val = {k: _safe_float(avg_raw.get(k)) for k in [
        "PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV",
        "FGM", "FGA", "FTM", "FTA",
    ]}

    out: Dict[str, float] = {}

    # Counting stats
    out["Z_PTS"]  = _z_scalar(val["PTS"],  LEAGUE_MEANS["PTS"],  LEAGUE_STDS["PTS"])
    out["Z_FG3M"] = _z_scalar(val["FG3M"], LEAGUE_MEANS["FG3M"], LEAGUE_STDS["FG3M"])
    out["Z_REB"]  = _z_scalar(val["REB"],  LEAGUE_MEANS["REB"],  LEAGUE_STDS["REB"])
    out["Z_AST"]  = _z_scalar(val["AST"],  LEAGUE_MEANS["AST"],  LEAGUE_STDS["AST"])
    out["Z_STL"]  = _z_scalar(val["STL"],  LEAGUE_MEANS["STL"],  LEAGUE_STDS["STL"])
    out["Z_BLK"]  = _z_scalar(val["BLK"],  LEAGUE_MEANS["BLK"],  LEAGUE_STDS["BLK"])
    # TOV: invert so that fewer turnovers => larger z
    out["Z_TOV"]  = _z_scalar(val["TOV"],  LEAGUE_MEANS["TOV"],  LEAGUE_STDS["TOV"], invert=True)

    # FG% impact: (FGM - league_fg_pct * FGA)
    if math.isnan(val["FGM"]) or math.isnan(val["FGA"]):
        out["Z_FG_PCT"] = float("nan")
    else:
        fg_impact = val["FGM"] - (LG_FG_PCT * val["FGA"])
        out["Z_FG_PCT"] = _z_scalar(fg_impact, FG_IMPACT_MEAN, FG_IMPACT_STD)

    # FT% impact: (FTM - league_ft_pct * FTA)
    if math.isnan(val["FTM"]) or math.isnan(val["FTA"]):
        out["Z_FT_PCT"] = float("nan")
    else:
        ft_impact = val["FTM"] - (LG_FT_PCT * val["FTA"])
        out["Z_FT_PCT"] = _z_scalar(ft_impact, FT_IMPACT_MEAN, FT_IMPACT_STD)

    # Average z across all components we just computed
    z_vals = [out[k] for k in out.keys() if k.startswith("Z_")]

    # If any component is NaN, AVG_Z should be NaN
    if not z_vals:
        out["AVG_Z"] = 0.0
    elif any(math.isnan(z) for z in z_vals):
        out["AVG_Z"] = float("nan")
    else:
        out["AVG_Z"] = sum(z_vals) / len(z_vals)

    return out
