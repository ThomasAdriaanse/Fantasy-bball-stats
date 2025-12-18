# app/blueprints/trades/routes.py
from __future__ import annotations
from typing import Dict, Any, List
import time
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import pandas as pd

from app.services.trade_pmf_eval import evaluate_trade_with_pmfs
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague

import compare_page.compare_page_data as cpd
from app.services.z_score_calculations import raw_to_zscore
from app.services import darko_services
from app.services.player_name_mapper import get_canonical_name

bp = Blueprint("trades", __name__)

CATEGORIES: List[str] = ["FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]
WINDOW_CHOICES: List[str] = ["total"]

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
        lg = (
            League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
            if espn_s2 and swid
            else League(league_id=league_id, year=year)
        )
        return lg, league_id, year
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied):
        return None, None, None
    except Exception:
        return None, None, None

# --------------------------- data builders ---------------------------
def _collect_league_players(use_darko_z: bool = False) -> pd.DataFrame:
    """
    Pull player data from cached volumes (season averages / DARKO).
    Returns DataFrame with:
      - player_name
      - team
      - raw stats
      - z_<cat> (either from Real Avgs or DARKO Projections)
    """
    # 1. Get the unified data (Real + DARKO keys)
    # This comes from cached JSON/CSV and does NOT engage the ESPN API/League object
    data_list = darko_services.get_darko_z_scores()

    rows: List[Dict[str, Any]] = []
    
    cat_to_zkey = {
        "FG%": "Z_FG",
        "FT%": "Z_FT",
        "3PM": "Z_FG3M",
        "REB": "Z_REB",
        "AST": "Z_AST",
        "STL": "Z_STL",
        "BLK": "Z_BLK",
        "TO":  "Z_TOV",
        "PTS": "Z_PTS",
    }
    
    for p in data_list:
        # Decide which source to use
        if use_darko_z:
            raw_d = p.get("RAW_DARKO", {})
            zd = p.get("Z_DARKO", {})
        else:
            raw_d = p.get("RAW_REAL", {}) or {}
            zd = p.get("Z_REAL", {}) or {}

        # Safe extraction
        def val(d, k): return float(d.get(k, 0.0))
        
        # Calculate percentages
        fgm, fga = val(raw_d, "FGM"), val(raw_d, "FGA")
        ftm, fta = val(raw_d, "FTM"), val(raw_d, "FTA")
        fg_pct = (fgm / fga) if fga > 0 else 0.0
        ft_pct = (ftm / fta) if fta > 0 else 0.0

        # Construct row
        row: Dict[str, Any] = {
            "player_name": p.get("player_name"),
            "team": p.get("team"), # NBA team
            "FG%": fg_pct,
            "FT%": ft_pct,
            "3PM": val(raw_d, "FG3M"),
            "REB": val(raw_d, "REB"),
            "AST": val(raw_d, "AST"),
            "STL": val(raw_d, "STL"),
            "BLK": val(raw_d, "BLK"),
            "TO":  val(raw_d, "TOV"),
            "PTS": val(raw_d, "PTS"),
        }
        
        # Attach Z-scores
        for cat, key in cat_to_zkey.items():
            row[f"z_{cat}"] = float(zd.get(key, 0.0))

        rows.append(row)
    
    if not rows:
        cols = ["player_name", "team"] + CATEGORIES + [f"z_{c}" for c in CATEGORIES]
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)


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
    """
    rows: List[Dict[str, Any]] = []
    
    for name in [n for n in names if n]:
        # Try to find player with name normalization
        canonical_name = get_canonical_name(name)
        
        # Match using various strategies
        rec = z_league[z_league["player_name"] == name]
        if rec.empty and canonical_name != name:
            rec = z_league[z_league["player_name"] == canonical_name]
        
        if rec.empty:
            # lower case attempt
            rec = z_league[z_league["player_name"].str.lower() == name.lower()]
            
        if rec.empty:
            cells = [{"cat": c, "z": 0.0, "raw": 0.0, "bg": _z_to_hsl_bg(0.0)} for c in CATEGORIES]
        else:
            r = rec.iloc[0]
            cells = []
            for c in CATEGORIES:
                z = float(r.get(f"z_{c}", 0.0))
                raw = float(r.get(c, 0.0))
                cells.append({"cat": c, "z": z, "raw": raw, "bg": _z_to_hsl_bg(z)})
        rows.append({"player_name": name, "cells": cells})
    return rows

# ----------------------------- routes ------------------------------
@bp.route("/trade", methods=["GET", "POST"])
def trade_analyzer():
    t0 = time.time()
    
    details = session.get("league_details") or {}
    league_id = details.get("league_id")
    year = details.get("year")
    
    if not league_id or not year:
        return redirect(url_for("main.entry_page", error_message="Enter your league first."))

    stat_window = "total"
    use_darko_z = request.form.get("use_darko_z") == "on"

    t1 = time.time()
    print(f"[TIMING] Session check: {t1 - t0:.4f}s")

    # 1. Load League - Directly from session (No Cache)
    league, _, _ = _get_league_from_session()
    if not league:
        return redirect(url_for("main.entry_page", error_message="Could not load league from ESPN."))

    t_league = time.time()
    print(f"[TIMING] League Load (ESPN API): {t_league - t1:.4f}s")

    # 2. Load Stats (Cached Disk)
    league_df = _collect_league_players(use_darko_z=use_darko_z)
    z_league = league_df if not league_df.empty else pd.DataFrame(
        columns=["player_name"] + CATEGORIES + [f"z_{c}" for c in CATEGORIES]
    )
    
    t2 = time.time()
    print(f"[TIMING] _collect_league_players (Stats): {t2 - t_league:.4f}s")
    
    # 3. Map Players to Fantasy Teams (Enrich DataFrame)
    player_team_map = {}
    if league:
        for team in league.teams:
            tid = getattr(team, 'team_id', 0)
            for p in team.roster:
                 player_team_map[p.name] = tid
                 
    # Add fantasy_team_id column
    if not league_df.empty:
         league_df["fantasy_team_id"] = league_df["player_name"].map(player_team_map).fillna(0).astype(int)

    # selections
    side_a = [request.values.get(k, "") for k in ("a1", "a2", "a3", "a4")]
    side_b = [request.values.get(k, "") for k in ("b1", "b2", "b3", "b4")]

    raw_team_a_id = request.values.get("team_a_idx")
    raw_team_b_id = request.values.get("team_b_idx")

    team_a_id = int(raw_team_a_id) if raw_team_a_id not in (None, "", "None") else None
    team_b_id = int(raw_team_b_id) if raw_team_b_id not in (None, "", "None") else None

    # Dropdowns: FANTASY TEAMS
    if league:
        team_options: List[Dict[str, Any]] = [
            {"idx": getattr(t, "team_id", idx), "name": t.team_name}
            for idx, t in enumerate(league.teams)
        ]
    else:
        team_options = []
    
    player_names = (
        sorted(league_df["player_name"].dropna().unique().tolist())
        if not league_df.empty else []
    )
    
    t3 = time.time()
    print(f"[TIMING] Dropdown building & Mapping: {t3 - t2:.4f}s")

    results = None
    pmf_result = None

    if request.method == "POST":
        t_post_start = time.time()
        
        if league:
            # ----- Z-scores -----
            a_tot = _sum_side_z(z_league, side_a)
            b_tot = _sum_side_z(z_league, side_b)
            diff = round(a_tot["_overall"] - b_tot["_overall"], 3)
            a_rows = _build_player_rows(z_league, side_a)
            b_rows = _build_player_rows(z_league, side_b)

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
            }
            
            t_z_calcs = time.time()
            print(f"[TIMING] Z-Score Calcs: {t_z_calcs - t_post_start:.4f}s")

            # ----- PMF Eval -----
            pmf_result = evaluate_trade_with_pmfs(
                league=league,
                year=year,
                stat_window=stat_window,
                side_a=side_a,
                side_b=side_b,
                team_a_idx=team_a_id, 
                team_b_idx=team_b_id,
                allowed_player_names=None, 
                use_darko_z=use_darko_z,
            )
            
            t_pmf = time.time()
            print(f"[TIMING] PMF Evaluation: {t_pmf - t_z_calcs:.4f}s")

    t_final = time.time()
    print(f"[TIMING] Total Route Time: {t_final - t0:.4f}s")
    
    return render_template(
        "trade_value.html",
        league_id=league_id,
        year=year,
        stat_window=stat_window,
        window_choices=WINDOW_CHOICES,
        player_names=player_names,
        side_a=side_a or ["", "", "", ""],
        side_b=side_b or ["", "", "", ""],
        categories=CATEGORIES,
        results=results,
        pmf_result=pmf_result,
        team_a_idx=team_a_id,
        team_b_idx=team_b_id,
        team_options=team_options,
        teams=team_options,
    )
