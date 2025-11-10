# app/services/percent_of_win_calculations.py

"""
Helpers for converting raw per-game stats into
'percent of a typical weekly win' units.

The idea:
- For counting stats (3PM, REB, AST, STL, BLK, TO, PTS),
  we express a player's per-game value as a percentage of a
  typical team weekly total in that category.

- For FG% and FT%, we scale the *difference* from league
  average by the player's volume (FGA / FTA).
"""

from __future__ import annotations
from typing import Dict

# Order we use throughout the app
CATS_ORDER = ["FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "PTS", "TO"]

# Average weekly team totals from last season
WEEKLY_TEAM_AVG: Dict[str, float] = {
    "3PM": 66.875,
    "REB": 221.9393939,
    "AST": 146.875,
    "STL": 39.40909091,
    "BLK": 24.59090909,
    "PTS": 620.6818182,
    "TO": 73.8030303,
}

# League-average percentages
LEAGUE_PERCENTS: Dict[str, float] = {
    "FG%": 0.473508333,
    "FT%": 0.79595,
}

# Average weekly attempts, used to weight FG% / FT% impact by volume
WEEKLY_ATTEMPTS: Dict[str, float] = {
    "FGA": 530.3333333,
    "FTA": 138.0833333,
}


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def raw_to_percent_of_win(avg_raw: Dict[str, float]) -> Dict[str, float]:
    """
    Convert per-game raw stats into estimated '% of a weekly win'
    for each scoring category.

    Input:
        avg_raw: per-game stats such as:
          {
            "FG%": 0.48, "FT%": 0.82,
            "3PM": 2.3, "REB": 7.1, "AST": 3.4,
            "STL": 1.1, "BLK": 0.8, "PTS": 18.5, "TO": 2.0,
            "FGA": 14.0, "FTA": 4.0,
          }

    Output:
        dict mapping category -> percent of weekly win, e.g.
          { "PTS": 1.25, "REB": 0.85, ... }
        where 1.00 means “~1% of a typical weekly win” in that category.
    """
    result: Dict[str, float] = {}

    # --- Percent categories: FG% and FT%, scaled by volume ---
    fg_pct = _safe_float(avg_raw.get("FG%"))
    ft_pct = _safe_float(avg_raw.get("FT%"))
    fga = _safe_float(avg_raw.get("FGA"))
    fta = _safe_float(avg_raw.get("FTA"))

    # FG% impact: difference from league FG%, scaled by FGA share
    if fga > 0 and WEEKLY_ATTEMPTS["FGA"] > 0:
        fg_diff = fg_pct - LEAGUE_PERCENTS["FG%"]
        fg_volume_share = fga / WEEKLY_ATTEMPTS["FGA"]
        # multiply by 100 so 0.01 => 1% of weekly win
        result["FG%"] = fg_diff * fg_volume_share * 100.0
    else:
        result["FG%"] = 0.0

    # FT% impact: difference from league FT%, scaled by FTA share
    if fta > 0 and WEEKLY_ATTEMPTS["FTA"] > 0:
        ft_diff = ft_pct - LEAGUE_PERCENTS["FT%"]
        ft_volume_share = fta / WEEKLY_ATTEMPTS["FTA"]
        result["FT%"] = ft_diff * ft_volume_share * 100.0
    else:
        result["FT%"] = 0.0

    # --- Counting categories: share of weekly total ---
    for cat in ["3PM", "REB", "AST", "STL", "BLK", "PTS", "TO"]:
        weekly = WEEKLY_TEAM_AVG.get(cat)
        if not weekly:
            result[cat] = 0.0
            continue
        val = _safe_float(avg_raw.get(cat))
        result[cat] = (val / weekly) * 100.0

    return result
