from __future__ import annotations
from typing import Dict, Any, List
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import pandas as pd

from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague

# reuse your compare helper to fetch per-team player rows by window
import compare_page.compare_page_data as cpd

bp = Blueprint("trades", __name__)

CATEGORIES: List[str] = ["FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]
WINDOW_CHOICES: List[str] = ["projected", "total", "last_30", "last_15", "last_7"]

# ----------------------- color helper -----------------------
def _z_to_hsl_bg(z: float, zmin: float = -1.6, zmax: float = 1.7) -> str:
    """
    Map z in [zmin, zmax] to a red→green HSL background with opacity.
    -1.6 => red (~0deg), 0 => yellow (~60deg), +1.7 => green (~120deg)
    """
    if z is None or np.isnan(z):
        return "transparent"
    t = (float(z) - zmin) / (zmax - zmin)  # 0..1
    t = max(0.0, min(1.0, t))
    hue = 120.0 * t
    # slightly darker so white text is readable
    return f"hsl({hue:.0f} 70% 35% / 0.90)"

# ----------------------- session/league helpers -----------------------
def _get_league_from_session():
    details = session.get("league_details") or {}
    league_id = details.get("league_id")
    year      = details.get("year")
    espn_s2   = details.get("espn_s2")
    swid      = details.get("swid")
    if not league_id or not year:
        return None, None, None
    try:
        lg = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid) if espn_s2 and swid \
             else League(league_id=league_id, year=year)
        return lg, league_id, year
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied):
        return None, None, None
    except Exception:
        return None, None, None

# --------------------------- data builders ---------------------------
def _collect_league_players(league, year: int, stat_window: str) -> pd.DataFrame:
    """
    Pull every team’s player rows for the given ESPN window and return a
    DataFrame with columns: player_name + the 9 cats (FG%/FT% derived from makes/attempts).
    """
    frames: List[pd.DataFrame] = []
    for team_idx, _ in enumerate(league.teams):
        df = cpd.get_team_player_data(
            league=league,
            team_num=team_idx,
            columns=[
                "player_name","min","fgm","fga","fg%","ftm","fta","ft%","threeptm",
                "reb","ast","stl","blk","turno","pts","inj","fpts","games"
            ],
            year=year,
            league_scoring_rules={  # neutralize fantasy scoring; we want raw 9-cat
                "fgm":0,"fga":0,"ftm":0,"fta":0,"threeptm":0,
                "reb":0,"ast":0,"stl":0,"blk":0,"turno":0,"pts":0
            },
            week_data=None,
            stat_window=stat_window
        )
        if df is not None and len(df):
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["player_name"] + CATEGORIES)

    df_all = pd.concat(frames, ignore_index=True)

    # numeric coercion
    for c in ["fgm","fga","ftm","fta","threeptm","reb","ast","stl","blk","turno","pts"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # derive more stable percentages from makes/attempts
    df_all["FG%"] = (df_all["fgm"] / df_all["fga"]).replace([np.inf, -np.inf], np.nan)
    df_all["FT%"] = (df_all["ftm"] / df_all["fta"]).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame({
        "player_name": df_all["player_name"],
        "FG%": df_all["FG%"].fillna(0.0),
        "FT%": df_all["FT%"].fillna(0.0),
        "3PM": df_all["threeptm"].fillna(0.0),
        "REB": df_all["reb"].fillna(0.0),
        "AST": df_all["ast"].fillna(0.0),
        "STL": df_all["stl"].fillna(0.0),
        "BLK": df_all["blk"].fillna(0.0),
        "TO":  df_all["turno"].fillna(0.0),
        "PTS": df_all["pts"].fillna(0.0),
    })

    # De-duplicate by player; keep row with highest PTS as a reasonable proxy
    out = out.sort_values(["player_name","PTS"], ascending=[True, False]) \
             .drop_duplicates("player_name", keep="first") \
             .reset_index(drop=True)

    return out

def _zscore_league(df_cat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leaguewide z for each category (TO inverted).
    Returns df with columns: player_name, 9 cats, and z_<cat> for each.
    """
    zdf = df_cat.copy()
    for cat in CATEGORIES:
        s = zdf[cat].astype(float)
        arr = -s.values if cat == "TO" else s.values  # invert TO so lower turnovers => higher value
        mu  = float(np.nanmean(arr))
        sd  = float(np.nanstd(arr)) or 1.0
        zdf[f"z_{cat}"] = (arr - mu) / sd
    return zdf

def _sum_side_z(zdf: pd.DataFrame, names: List[str]) -> Dict[str, float]:
    """
    Sum z per category across the given players; also return an '_overall' sum.
    """
    want = [n for n in names if n]
    if not want:
        totals = {cat: 0.0 for cat in CATEGORIES}
        totals["_overall"] = 0.0
        return totals

    pick = zdf[zdf["player_name"].isin(want)]
    totals = {cat: round(float(np.nansum(pick[f"z_{cat}"].values)), 3) for cat in CATEGORIES}
    totals["_overall"] = round(sum(totals[c] for c in CATEGORIES), 3)
    return totals

def _build_player_rows(z_league: pd.DataFrame, names: List[str]) -> List[Dict[str, Any]]:
    """
    Build per-player rows preserving selection order.
    Each row: {'player_name': str, 'cells': [{'cat': c, 'z': float, 'bg': str}, ...]}
    """
    rows: List[Dict[str, Any]] = []
    for name in [n for n in names if n]:
        rec = z_league[z_league["player_name"] == name]
        if rec.empty:
            cells = [{"cat": c, "z": 0.0, "bg": _z_to_hsl_bg(0.0)} for c in CATEGORIES]
        else:
            r = rec.iloc[0]
            cells = []
            for c in CATEGORIES:
                z = float(r.get(f"z_{c}", 0.0))
                cells.append({"cat": c, "z": z, "bg": _z_to_hsl_bg(z)})
        rows.append({"player_name": name, "cells": cells})
    return rows

# ----------------------------- routes ------------------------------
@bp.route("/trade", methods=["GET", "POST"])
def trade_analyzer():
    league, league_id, year = _get_league_from_session()
    if league is None:
        return redirect(url_for("main.entry_page", error_message="Enter your league first."))

    stat_window = (request.values.get("window") or "projected").strip().lower()
    if stat_window not in WINDOW_CHOICES:
        stat_window = "projected"

    # Build league pool for this window and compute z
    league_df = _collect_league_players(league, year, stat_window)
    z_league  = _zscore_league(league_df) if not league_df.empty else pd.DataFrame(
        columns=["player_name"] + CATEGORIES + [f"z_{c}" for c in CATEGORIES]
    )

    # selections (up to 4 each side)
    side_a = [request.values.get(k, "") for k in ("a1","a2","a3","a4")]
    side_b = [request.values.get(k, "") for k in ("b1","b2","b3","b4")]

    # dropdown options
    player_names = sorted(league_df["player_name"].dropna().unique().tolist()) if not league_df.empty else []

    results = None
    if request.method == "POST":
        a_tot = _sum_side_z(z_league, side_a)
        b_tot = _sum_side_z(z_league, side_b)
        diff  = round(a_tot["_overall"] - b_tot["_overall"], 3)

        # per-player contributions for the tables (with backend colors)
        a_rows = _build_player_rows(z_league, side_a)
        b_rows = _build_player_rows(z_league, side_b)

        # ΔZ row (A - B) with backend-colored cells
        delta_cells = []
        for c in CATEGORIES:
            z = round(float(a_tot.get(c, 0.0) - b_tot.get(c, 0.0)), 2)
            delta_cells.append({"cat": c, "z": z, "bg": _z_to_hsl_bg(z)})
        delta_overall = round(a_tot["_overall"] - b_tot["_overall"], 2)
        delta_overall_bg = _z_to_hsl_bg(delta_overall)

        results = {
            "window": stat_window,
            "a_totals": a_tot,
            "b_totals": b_tot,
            "a_players_rows": a_rows,
            "b_players_rows": b_rows,
            "delta_cells": delta_cells,
            "delta_overall": delta_overall,
            "delta_overall_bg": delta_overall_bg,
            "diff": diff,
            # verdict removed by request; keep diff for potential future use
        }

    return render_template(
        "trade_value.html",
        league_id=league_id,
        year=year,
        stat_window=stat_window,
        window_choices=WINDOW_CHOICES,
        player_names=player_names,
        side_a=side_a or ["","","",""],
        side_b=side_b or ["","","",""],
        categories=CATEGORIES,
        results=results
    )

# Optional JSON API if you later want dynamic updates
@bp.post("/trade/api")
def trade_api():
    league, _, year = _get_league_from_session()
    if league is None:
        return jsonify({"error": "No league in session"}), 400

    data = request.get_json(force=True, silent=True) or {}
    window = (data.get("window") or "projected").strip().lower()
    if window not in WINDOW_CHOICES:
        window = "projected"

    league_df = _collect_league_players(league, year, window)
    z_league  = _zscore_league(league_df) if not league_df.empty else pd.DataFrame(
        columns=["player_name"] + CATEGORIES + [f"z_{c}" for c in CATEGORIES]
    )

    side_a = (data.get("side_a") or [])[:4]
    side_b = (data.get("side_b") or [])[:4]

    a_tot = _sum_side_z(z_league, side_a)
    b_tot = _sum_side_z(z_league, side_b)
    diff  = round(a_tot["_overall"] - b_tot["_overall"], 3)

    # ΔZ cells & overall (with colors)
    delta_cells = []
    for c in CATEGORIES:
        z = round(float(a_tot.get(c, 0.0) - b_tot.get(c, 0.0)), 2)
        delta_cells.append({"cat": c, "z": z, "bg": _z_to_hsl_bg(z)})
    delta_overall = round(a_tot["_overall"] - b_tot["_overall"], 2)
    delta_overall_bg = _z_to_hsl_bg(delta_overall)

    return jsonify({
        "categories": CATEGORIES,
        "window": window,
        "a_totals": a_tot,
        "b_totals": b_tot,
        "delta_cells": delta_cells,
        "delta_overall": delta_overall,
        "delta_overall_bg": delta_overall_bg,
        "diff": diff
    })
