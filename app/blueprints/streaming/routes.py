# app/blueprints/streaming/routes.py
from __future__ import annotations

import json
import math
from flask import Blueprint, render_template, session, redirect, url_for, request
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague

from app.services.percent_of_win_calculations import (
    CATS_ORDER,   # still re-use the category order: ["FG%", "FT%", "3PM", ...]
)
from app.services.z_score_calculations import raw_to_zscore

bp = Blueprint("streaming", __name__)


def _parse_json_field(form, name):
    raw = form.get(name, "")
    if not raw:
        return None, f"{name}=<empty>"
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return val, f"{name}=list(len={len(val)})"
        if isinstance(val, dict):
            return val, f"{name}=dict(keys={list(val.keys())[:8]})"
        return val, f"{name}={type(val).__name__}"
    except Exception:
        return None, f"{name}=<invalid json> (len={len(raw)})"


@bp.post("/")
def streaming_page():
    print("HIT /streaming (POST)")
    print("  session.league_details =", session.get("league_details"))
    print("  form.keys() =", list(request.form.keys()))
    print("  args =", request.args.to_dict())

    league_details = session.get("league_details") or {}

    # --- build League (public vs private) ---
    try:
        league_id = int(league_details["league_id"])
        year = int(league_details["year"])
        if league_details.get("espn_s2") and league_details.get("swid"):
            league = League(
                league_id=league_id,
                year=year,
                espn_s2=league_details["espn_s2"],
                swid=league_details["swid"],
            )
        else:
            league = League(league_id=league_id, year=year)
    except (ESPNUnknownError, ESPNInvalidLeague):
        return redirect(url_for("main.entry_page", error_message="Invalid league entered."))
    except ESPNAccessDenied:
        return redirect(
            url_for("main.entry_page", error_message="That is a private league which needs ESPN S2 and SWID.")
        )
    except Exception as e:
        return redirect(url_for("main.entry_page", error_message=str(e)))

    # --- posted form fields (optional) ---
    my_team = request.form.get("myTeam", "My Team")
    opp_team = request.form.get("opponentsTeam", "Opponent")
    stat_window = request.form.get("stat_window", "")
    scoring_type = request.form.get("scoring_type", "")

    week_data, _ = _parse_json_field(request.form, "week_data_json")
    team1_win_pct, _ = _parse_json_field(request.form, "team1_win_pct_json")
    team2_win_pct, _ = _parse_json_field(request.form, "team2_win_pct_json")

    # derive a week number if present (for template display)
    week_num = None
    if isinstance(week_data, dict):
        week_num = week_data.get("selected_week")
        if week_num is None and isinstance(week_data.get("matchup"), dict):
            week_num = week_data["matchup"].get("selected_week")

    # ================================
    # Weights from win% odds (1 at 50/50 → 0 at 0/100)
    # ================================
    PAYLOAD_TO_DISPLAY = {
        "fg%": "FG%",
        "ft%": "FT%",
        "threeptm": "3PM",
        "reb": "REB",
        "ast": "AST",
        "stl": "STL",
        "blk": "BLK",
        "pts": "PTS",
        "turno": "TO",
    }
    DISPLAY_TO_PAYLOAD = {v: k for k, v in PAYLOAD_TO_DISPLAY.items()}

    # Expect [ { ... } ]
    t1 = team1_win_pct[0] if (
        isinstance(team1_win_pct, list)
        and len(team1_win_pct) == 1
        and isinstance(team1_win_pct[0], dict)
    ) else None
    t2 = team2_win_pct[0] if (
        isinstance(team2_win_pct, list)
        and len(team2_win_pct) == 1
        and isinstance(team2_win_pct[0], dict)
    ) else None

    if t1 is None and t2 is None:
        weights_by_cat = {c: 0.0 for c in CATS_ORDER}
    else:
        weights_by_cat = {}
        for cat in CATS_ORDER:
            pk = DISPLAY_TO_PAYLOAD.get(cat)
            if not pk:
                continue
            p = None
            if t1 is not None and isinstance(t1.get(pk), (int, float)):
                p = float(t1[pk]) / 100.0
            elif t2 is not None and isinstance(t2.get(pk), (int, float)):
                p = 1.0 - (float(t2[pk]) / 100.0)
            if p is None:
                continue
            p = max(0.0, min(1.0, p))
            # weight: 1 at 50/50, 0 at 0 or 1
            weights_by_cat[cat] = max(0.0, 1.0 - 2.0 * abs(p - 0.5))

    # ================================
    # Free agents → raw stats + z-scores
    # ================================
    def _avg(stats_avg: dict, k: str) -> float:
        try:
            return float((stats_avg or {}).get(k, 0.0))
        except (TypeError, ValueError):
            return 0.0

    year_key = f"{year}_total"

    # Injury filter: exclude OUT / INJ (and similar short codes)
    def _is_injured(p) -> bool:
        status = getattr(p, "injuryStatus", "") or getattr(p, "injury_status", "")
        status = str(status).strip().upper()
        return status in {"OUT", "INJ", "O", "IL", "IR"}

    free_agents_raw = league.free_agents(size=200)
    free_agents = [fa for fa in free_agents_raw if not _is_injured(fa)]

    candidates = []
    for p in free_agents:
        stats_block = getattr(p, "stats", {}) or {}
        stat_year = stats_block.get(year_key) or {}
        avg = stat_year.get("avg") or {}

        # RAW per-game values (for raw view)
        raw_fg_pct = _avg(avg, "FG%")
        raw_ft_pct = _avg(avg, "FT%")
        raw_3pm    = _avg(avg, "3PM")
        raw_reb    = _avg(avg, "REB")
        raw_ast    = _avg(avg, "AST")
        raw_stl    = _avg(avg, "STL")
        raw_blk    = _avg(avg, "BLK")
        raw_pts    = _avg(avg, "PTS")
        raw_to     = _avg(avg, "TO")

        # Extra fields needed for z-score math
        fgm = _avg(avg, "FGM")
        fga = _avg(avg, "FGA")
        ftm = _avg(avg, "FTM")
        fta = _avg(avg, "FTA")

        # Build input for z-scores helper
        avg_raw_for_z = {
            "PTS":  raw_pts,
            "FG3M": raw_3pm,   # ESPN "3PM" → z-score "FG3M"
            "REB":  raw_reb,
            "AST":  raw_ast,
            "STL":  raw_stl,
            "BLK":  raw_blk,
            "TOV":  raw_to,
            "FGM":  fgm,
            "FGA":  fga,
            "FTM":  ftm,
            "FTA":  fta,
        }
        z_stats = raw_to_zscore(avg_raw_for_z) or {}

        # Map z-scores back into your 9-cat labels
        z_by_cat = {
            "FG%": z_stats.get("Z_FG", 0.0),
            "FT%": z_stats.get("Z_FT", 0.0),
            "3PM": z_stats.get("Z_FG3M",   0.0),
            "REB": z_stats.get("Z_REB",    0.0),
            "AST": z_stats.get("Z_AST",    0.0),
            "STL": z_stats.get("Z_STL",    0.0),
            "BLK": z_stats.get("Z_BLK",    0.0),
            "PTS": z_stats.get("Z_PTS",    0.0),
            "TO":  z_stats.get("Z_TOV",    0.0),
        }

        # Scores: unweighted vs weighted (by matchup odds)
        score_unweighted = 0.0
        score_weighted = 0.0
        for cat in CATS_ORDER:
            val = z_by_cat.get(cat, 0.0)
            if isinstance(val, float) and math.isnan(val):
                continue
            w = weights_by_cat.get(cat, 0.0)
            score_unweighted += val
            score_weighted   += val * w

        candidates.append({
            "player": getattr(p, "name", str(p)),
            "team": (
                getattr(p, "proTeam", None)
                or getattr(p, "team", None)
                or getattr(p, "team_name", None)
                or ""
            ),
            "pos": getattr(p, "position", None) or getattr(p, "pos", None) or "",
            "games_str": "",  # still empty for now, can fill later

            # per-cat z-scores and raw per-game stats
            "avg_z": {c: float(z_by_cat.get(c, 0.0)) for c in CATS_ORDER},
            "avg_raw": {
                "FG%": raw_fg_pct,
                "FT%": raw_ft_pct,
                "3PM": raw_3pm,
                "REB": raw_reb,
                "AST": raw_ast,
                "STL": raw_stl,
                "BLK": raw_blk,
                "PTS": raw_pts,
                "TO":  raw_to,
            },
            "score_weighted": score_weighted,
            "score_unweighted": score_unweighted,
        })

    # Sort by weighted score, take top 10
    top10 = sorted(candidates, key=lambda x: x.get("score_weighted", 0.0), reverse=True)[:10]

    # Optional text describing weights
    parts = [f"{c}: {weights_by_cat.get(c,0.0):.2f}" for c in CATS_ORDER]
    modifiers_text = (
        "Toggle between z-scores and raw per-game. "
        "Weights emphasize categories closest to 50/50 in your matchup. "
        + ", ".join(parts)
    )

    # Placeholder streaming plan (you can compute this later)
    stream_plan = []
    remaining_moves = 0

    return render_template(
        "streaming.html",
        my_team=my_team,
        opp_team=opp_team,
        week_num=week_num,
        top10=top10,
        stream_plan=stream_plan,
        remaining_moves=remaining_moves,
        modifiers_text=modifiers_text,
        weights_by_cat=weights_by_cat,
        cats_order=CATS_ORDER,
    )
