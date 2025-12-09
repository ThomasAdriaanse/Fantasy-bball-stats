# app/blueprints/trades/routes.py
from __future__ import annotations
from typing import Dict, Any, List
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import pandas as pd

from app.services.trade_pmf_eval import evaluate_trade_with_pmfs
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague

import compare_page.compare_page_data as cpd
from app.services.z_score_calculations import raw_to_zscore

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
def _collect_league_players(league, year: int, stat_window: str) -> pd.DataFrame:
    """
    Pull every team’s player rows for the given ESPN window and return a
    DataFrame with:
      - player_name
      - raw 9-cat stats (FG%, FT%, 3PM, REB, AST, STK, BLK, TO, PTS)
      - z_<cat> per category using raw_to_zscore (NBA-wide baselines).
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
            league_scoring_rules={  # neutralize fantasy scoring; just want raw stats
                "fgm":0,"fga":0,"ftm":0,"fta":0,"threeptm":0,
                "reb":0,"ast":0,"stl":0,"blk":0,"turno":0,"pts":0
            },
            week_data=None,
            stat_window=stat_window
        )
        if df is not None and len(df):
            frames.append(df)

    if not frames:
        cols = ["player_name"] + CATEGORIES + [f"z_{c}" for c in CATEGORIES]
        return pd.DataFrame(columns=cols)

    df_all = pd.concat(frames, ignore_index=True)

    # numeric coercion for the base stats we use
    for c in ["fgm","fga","ftm","fta","threeptm","reb","ast","stl","blk","turno","pts"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0.0)

    # Raw per-game frame in the shape raw_to_zscore expects
    df_raw = pd.DataFrame({
        "player_name": df_all["player_name"],
        "FGM": df_all["fgm"],
        "FGA": df_all["fga"],
        "FTM": df_all["ftm"],
        "FTA": df_all["fta"],
        "FG3M": df_all["threeptm"],
        "REB": df_all["reb"],
        "AST": df_all["ast"],
        "STL": df_all["stl"],
        "BLK": df_all["blk"],
        "TOV": df_all["turno"],
        "PTS": df_all["pts"],
    })

    # De-duplicate by player; keep row with highest PTS as a reasonable proxy
    df_raw = (
        df_raw
        .sort_values(["player_name", "PTS"], ascending=[True, False])
        .drop_duplicates("player_name", keep="first")
        .reset_index(drop=True)
    )

    rows: List[Dict[str, Any]] = []

    # Map z-score keys -> 9-cat names
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

    for _, r in df_raw.iterrows():
        # Raw 9-cat values
        fgm = float(r["FGM"])
        fga = float(r["FGA"])
        ftm = float(r["FTM"])
        fta = float(r["FTA"])

        fg_pct = (fgm / fga) if fga > 0 else 0.0
        ft_pct = (ftm / fta) if fta > 0 else 0.0

        raw_for_z = {
            "PTS":  float(r["PTS"]),
            "FG3M": float(r["FG3M"]),
            "REB":  float(r["REB"]),
            "AST":  float(r["AST"]),
            "STL":  float(r["STL"]),
            "BLK":  float(r["BLK"]),
            "TOV":  float(r["TOV"]),
            "FGM":  fgm,
            "FGA":  fga,
            "FTM":  ftm,
            "FTA":  fta,
        }

        zmap = raw_to_zscore(raw_for_z)

        row: Dict[str, Any] = {
            "player_name": r["player_name"],
            "FG%": fg_pct,
            "FT%": ft_pct,
            "3PM": float(r["FG3M"]),
            "REB": float(r["REB"]),
            "AST": float(r["AST"]),
            "STL": float(r["STL"]),
            "BLK": float(r["BLK"]),
            "TO":  float(r["TOV"]),
            "PTS": float(r["PTS"]),
        }

        # Attach per-category z_<cat> columns
        for cat, key in cat_to_zkey.items():
            row[f"z_{cat}"] = float(zmap.get(key, 0.0))

        rows.append(row)

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

    # Debug: show mapping of enumerate index -> team_id -> name
    print("[TRADES] League teams (idx -> team_id -> name):")
    for idx, t in enumerate(league.teams):
        print(f"  idx={idx}  team_id={getattr(t, 'team_id', None)}  name={t.team_name}")

    stat_window = (request.values.get("window") or "total").strip().lower()
    if stat_window not in WINDOW_CHOICES:
        stat_window = "total"

    # Build league pool for this window, including z_<cat> columns
    league_df = _collect_league_players(league, year, stat_window)
    z_league = league_df if not league_df.empty else pd.DataFrame(
        columns=["player_name"] + CATEGORIES + [f"z_{c}" for c in CATEGORIES]
    )

    # selections (up to 4 each side)
    side_a = [request.values.get(k, "") for k in ("a1", "a2", "a3", "a4")]
    side_b = [request.values.get(k, "") for k in ("b1", "b2", "b3", "b4")]

    # team selectors from UI (use team_id as canonical)
    raw_team_a_id = request.values.get("team_a_idx")
    raw_team_b_id = request.values.get("team_b_idx")

    team_a_id = int(raw_team_a_id) if raw_team_a_id not in (None, "", "None") else None
    team_b_id = int(raw_team_b_id) if raw_team_b_id not in (None, "", "None") else None

    # ---------- Build team options for the dropdowns ----------
    # Use team.team_id as the ID exposed to the UI
    team_options: List[Dict[str, Any]] = [
        {"idx": getattr(t, "team_id", idx), "name": t.team_name}
        for idx, t in enumerate(league.teams)
    ]
    print("[TRADES] team_options (team_id, name):", [(t["idx"], t["name"]) for t in team_options])

    # dropdown options for players
    player_names = (
        sorted(league_df["player_name"].dropna().unique().tolist())
        if not league_df.empty else []
    )

    results = None
    pmf_result = None

    if request.method == "POST":
        print(
            f"[TRADES] POST /trade "
            f"team_a_id={team_a_id} team_b_id={team_b_id} "
            f"side_a={side_a} side_b={side_b} window={stat_window}"
        )

        # ----- Z-score based trade table -----
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

        # ----- Top 10 Filter Logic -----
        top_10_only = request.form.get("top_10_only") == "on"
        allowed_player_names = None

        if top_10_only:
            # We need z-scores for ALL players to determine top 10 per team.
            # _collect_league_players returns a DF with "z_<cat>" columns.
            # We can sum them to get "Total Z".
            
            # Calculate total Z for ranking
            z_cols = [f"z_{c}" for c in CATEGORIES]
            # Ensure columns exist (fill 0 if missing)
            for zc in z_cols:
                if zc not in league_df.columns:
                    league_df[zc] = 0.0
                    print("error adding column: ", zc)
            
            league_df["_total_z"] = league_df[z_cols].sum(axis=1)
            
            # We need to map players to teams to filter per-team.
            # league_df doesn't explicitly have team_id, but we can infer or re-fetch.
            # Actually, _collect_league_players iterates teams. Let's just re-iterate league.teams
            # and match names, or better yet, just use the fact that we have the roster objects.
            
            allowed_player_names = set()
            
            # Iterate through all teams to find their top 10
            for team in league.teams:
                team_player_names = [p.name for p in team.roster]
                # Filter DF for this team
                team_df = league_df[league_df["player_name"].isin(team_player_names)]
                
                if not team_df.empty:
                    # Sort by Total Z descending
                    top_10 = team_df.sort_values("_total_z", ascending=False).head(10)
                    allowed_player_names.update(top_10["player_name"].tolist())
            
            print(f"[TRADES] Top 10 filter active. Allowed {len(allowed_player_names)} players.")

        # ----- PMF-based trade evaluation (before/after, all teams) -----
        # We now pass team_id values into evaluate_trade_with_pmfs
        pmf_result = evaluate_trade_with_pmfs(
            league=league,
            year=year,
            stat_window=stat_window,
            side_a=side_a,
            side_b=side_b,
            team_a_idx=team_a_id,   # interpreted as team_id on the PMF side
            team_b_idx=team_b_id,
            allowed_player_names=allowed_player_names,
        )
        
        if pmf_result:
            #print("\n[TRADES] --- Trade Evaluation Results ---")
            tt = pmf_result.get("trading_teams", {})
            ta_id = tt.get("team_a_idx")
            tb_id = tt.get("team_b_idx")
            ta_name = tt.get("team_a_name")
            tb_name = tt.get("team_b_name")
            
            #print(f"Trade between: {ta_name} (ID: {ta_id}) and {tb_name} (ID: {tb_id})")
            
            before_avg = pmf_result.get("before", {}).get("avg_win_pct", {})
            after_avg = pmf_result.get("after", {}).get("avg_win_pct", {})
            
            #print(f"\n{ta_name} Average Win %:")
            #print(f"  Before: {before_avg.get(ta_id)}")
            #print(f"  After:  {after_avg.get(ta_id)}")
            
            #print(f"\n{tb_name} Average Win %:")
            #print(f"  Before: {before_avg.get(tb_id)}")
            #print(f"  After:  {after_avg.get(tb_id)}")
            #print("---------------------------------------\n")
        else:
            print(" ")
            #print("[TRADES] PMF result is None (could not simulate trade).")

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
        team_options=team_options,   # for dropdowns
        teams=team_options,          # alias, in case template still references `teams`
    )
