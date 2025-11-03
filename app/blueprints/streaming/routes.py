from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, session
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Tuple
import json

from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague

bp = Blueprint("streaming", __name__)

def _date_only(dt) -> date:
    return (dt - timedelta(hours=5)).date()

def _week_dates(week_data: Dict[str, Any]) -> List[date]:
    import pandas as pd
    sd = datetime.strptime(week_data["matchup_data"]["start_date"], "%Y-%m-%d").date()
    ed = datetime.strptime(week_data["matchup_data"]["end_date"],   "%Y-%m-%d").date()
    return pd.date_range(start=sd, end=ed + timedelta(days=1)).date.tolist()

def _row_to_avg_map(row: Dict[str, Any]) -> Dict[str, float]:
    def f(x):
        try:
            return 0.0 if x in (None, "N/A") else float(x)
        except:
            return 0.0
    return {
        "3PM": f(row.get("threeptm")),
        "REB": f(row.get("reb")),
        "AST": f(row.get("ast")),
        "STL": f(row.get("stl")),
        "BLK": f(row.get("blk")),
        "PTS": f(row.get("pts")),
        "TO":  f(row.get("turno")),
    }

def _fa_avg_map(p) -> Dict[str, float]:
    stats_all = getattr(p, "stats", {}) or {}
    keys = list(stats_all.keys())
    prio = [k for k in keys if k.endswith("_projected")] + \
           [k for k in keys if k.endswith("_last_30")] + \
           [k for k in keys if k.endswith("_total")]
    avg = {}
    for k in prio:
        blk = stats_all.get(k) or {}
        if isinstance(blk, dict) and isinstance(blk.get("avg"), dict) and blk["avg"]:
            avg = blk["avg"]; break
    def pick(*names):
        for n in names:
            if n in avg and avg[n] is not None:
                try: return float(avg[n])
                except: return 0.0
        return 0.0
    return {
        "3PM": pick("3PM","TPM"), "REB": pick("REB","RPG"), "AST": pick("AST","APG"),
        "STL": pick("STL","SPG"), "BLK": pick("BLK","BPG"), "PTS": pick("PTS","PPG"),
        "TO":  pick("TO","TOPG"),
    }

@bp.post("/")
def streaming_page():
    # Must be reached from compare page (POST)
    details = session.get("league_details") or {}
    if not details or not details.get("league_id") or not details.get("year"):
        return redirect(url_for('main.entry_page', error_message="Enter league info first."))

    # --- unpack payload from compare page (strings -> json) ---
    my_team_name        = request.form.get("myTeam")
    opponents_team_name = request.form.get("opponentsTeam")
    week_data           = json.loads(request.form.get("week_data_json") or "{}")
    stat_window         = (request.form.get("stat_window") or "projected")
    scoring_type        = (request.form.get("scoring_type") or "H2H_CATEGORY")

    team1_rows          = json.loads(request.form.get("team1_rows_json") or "[]")
    team2_rows          = json.loads(request.form.get("team2_rows_json") or "[]")
    win1_rows           = json.loads(request.form.get("team1_win_pct_json") or "[]")
    win2_rows           = json.loads(request.form.get("team2_win_pct_json") or "[]")
    cur1                = json.loads(request.form.get("team1_current_stats_json") or "{}")
    cur2                = json.loads(request.form.get("team2_current_stats_json") or "{}")

    if not week_data or "matchup_data" not in week_data:
        return redirect(url_for('compare.select_teams_page'))

    # --- derive swing categories ONLY from win1 (already computed) ---
    swing_cats: List[str] = []
    if win1_rows:
        row = dict(win1_rows[0])  # {'pts': %, 'reb': %, 'threeptm': %, 'turno': %, 'fg%': %, 'ft%': %}
        key_map = {"threeptm": "3PM", "fg%": "FG%", "ft%": "FT%", "turno": "TO"}
        for k, v in row.items():
            try:
                p = float(v)
            except:
                continue
            disp = key_map.get(k.lower(), k.upper())
            if disp in ("3PM","REB","AST","STL","BLK","PTS","TO") and 40.0 <= p <= 60.0:
                swing_cats.append(disp)
    if not swing_cats:
        swing_cats = ["STL","BLK","3PM"]  # tiny fallback

    # --- roster maps from team1_rows (no schedule info) ---
    roster_avg = {r["player_name"]: _row_to_avg_map(r) for r in team1_rows}

    # --- week dates from payload ---
    week_dates = _week_dates(week_data)
    today_cut = (datetime.today() - timedelta(hours=8)).date()

    # --- build league JUST to read free agents (waiver pool + their schedules) ---
    try:
        if details.get("espn_s2") and details.get("swid"):
            league = League(league_id=details["league_id"], year=details["year"],
                            espn_s2=details["espn_s2"], swid=details["swid"])
        else:
            league = League(league_id=details["league_id"], year=details["year"])
    except (ESPNUnknownError, ESPNInvalidLeague):
        return redirect(url_for('main.entry_page', error_message="Invalid league entered."))
    except ESPNAccessDenied:
        return redirect(url_for('main.entry_page', error_message="Private league. Provide ESPN S2 + SWID."))
    except Exception as e:
        return redirect(url_for('main.entry_page', error_message=str(e)))

    fa_pool = league.free_agents(size=250)

    # --- minimal per-day plan: choose best FA playing that date; drop lowest-contrib player (no roster schedule) ---
    plan: List[Dict[str, Any]] = []
    for d in week_dates:
        if d < today_cut:
            continue

        # FA playing this date scored by swing cats sum
        best_add = None; best_add_score = -1.0
        for fa in fa_pool:
            sched = [_date_only(g["date"]) for g in (getattr(fa, "schedule", {}) or {}).values()]
            if d not in sched:
                continue
            fa_avg = _fa_avg_map(fa)
            score = sum(fa_avg.get(c, 0.0) for c in swing_cats)
            if score > best_add_score:
                best_add_score, best_add = score, fa

        if not best_add:
            continue

        # Drop = smallest swing sum among current roster rows
        best_drop_name = None; best_drop_score = 10**9
        for name, avg in roster_avg.items():
            score = sum(avg.get(c, 0.0) for c in swing_cats)
            if score < best_drop_score:
                best_drop_score, best_drop_name = score, name

        if not best_drop_name:
            continue

        add_avg  = _fa_avg_map(best_add)
        drop_avg = roster_avg.get(best_drop_name, {"3PM":0,"REB":0,"AST":0,"STL":0,"BLK":0,"PTS":0,"TO":0})
        delta_preview = {c: round(add_avg.get(c,0.0) - drop_avg.get(c,0.0), 2) for c in swing_cats}

        plan.append({
            "date": d.strftime("%Y-%m-%d"),
            "add_player": getattr(best_add, "name", "Unknown"),
            "add_team": getattr(best_add, "proTeam", ""),
            "add_pos": getattr(best_add, "position", ""),
            "drop_player": best_drop_name,
            "targets": swing_cats,
            "delta_preview": delta_preview
        })

    return render_template(
        "streaming.html",
        my_team=my_team_name,
        opp_team=opponents_team_name,
        week_num=week_data.get("selected_week"),
        swing_cats=swing_cats,
        plan=plan
    )
