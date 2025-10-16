from __future__ import annotations
from flask import Blueprint, render_template, request, jsonify
from typing import Dict, List, Tuple
import pandas as pd

# reuse existing helpers you already have
from app.services.player_stats import get_active_players_list
from app.services.s3_service import load_player_dataset_from_s3

bp = Blueprint("trades", __name__)

Z_CATS = [
    "Z_PTS", "Z_FG3M", "Z_REB", "Z_AST", "Z_STL", "Z_BLK",
    "Z_FG_PCT", "Z_FT_PCT", "Z_TOV"
]

def _mean_last_n(series: pd.Series, n: int) -> float:
    if series is None or series.empty:
        return 0.0
    if n and n > 0:
        s = pd.to_numeric(series.tail(n), errors="coerce")
    else:
        s = pd.to_numeric(series, errors="coerce")
    val = float(s.mean()) if s.notna().any() else 0.0
    return round(val, 3)

def _player_z_summary(player_name: str, num_games: int) -> Dict[str, float]:
    """
    Load one player's dataset from S3 and return averaged Z-cats over the last N games.
    If a Z_* column is missing, treat as 0.0 (neutral).
    """
    df = load_player_dataset_from_s3(player_name)
    if df is None or df.empty:
        # return zeroed profile so page doesn't explode
        return {c: 0.0 for c in Z_CATS}

    # Ensure the most-recent first/last ordering
    if "Game_Number" in df.columns:
        df = df.sort_values("Game_Number").reset_index(drop=True)

    result = {}
    for c in Z_CATS:
        if c in df.columns:
            result[c] = _mean_last_n(df[c], num_games)
        else:
            # Missing z-column: neutral
            result[c] = 0.0
    return result

def _sum_side(players: List[str], num_games: int) -> Dict[str, float]:
    totals = {c: 0.0 for c in Z_CATS}
    for name in players:
        if not name:
            continue
        zmap = _player_z_summary(name, num_games)
        for c in Z_CATS:
            totals[c] += zmap.get(c, 0.0)
    # round for display
    return {k: round(v, 3) for k, v in totals.items()}

def _overall_score(cat_totals: Dict[str, float]) -> float:
    # simple sum of all z-categories (note: Z_TOV is already negative for high TO)
    return round(sum(cat_totals.get(c, 0.0) for c in Z_CATS), 3)

@bp.route("/trade", methods=["GET", "POST"])
def trade_analyzer():
    players = get_active_players_list()  # [{full_name: ...}, ...]
    player_names = [p.get("full_name", "") for p in players if p.get("full_name")]

    # defaults
    side_a = ["", "", "", ""]
    side_b = ["", "", "", ""]
    num_games = 20  # last-N window for Z averages (0 -> full season)

    results = None

    if request.method == "POST":
        side_a = [
            request.form.get("a1", "").strip(),
            request.form.get("a2", "").strip(),
            request.form.get("a3", "").strip(),
            request.form.get("a4", "").strip(),
        ]
        side_b = [
            request.form.get("b1", "").strip(),
            request.form.get("b2", "").strip(),
            request.form.get("b3", "").strip(),
            request.form.get("b4", "").strip(),
        ]
        try:
            num_games = int(request.form.get("num_games", num_games))
        except ValueError:
            num_games = 20

        a_totals = _sum_side([n for n in side_a if n], num_games)
        b_totals = _sum_side([n for n in side_b if n], num_games)
        a_score  = _overall_score(a_totals)
        b_score  = _overall_score(b_totals)
        diff     = round(a_score - b_score, 3)

        verdict  = "Side A looks better" if diff > 0 else ("Side B looks better" if diff < 0 else "Even trade")

        results = {
            "a_totals": a_totals,
            "b_totals": b_totals,
            "a_score": a_score,
            "b_score": b_score,
            "diff": diff,
            "verdict": verdict
        }

    return render_template(
        "trade_value.html",
        player_names=player_names,
        z_categories=Z_CATS,
        side_a=side_a,
        side_b=side_b,
        num_games=num_games,
        results=results
    )

# Optional: JSON API
@bp.post("/api/trade_value")
def trade_value_api():
    payload = request.get_json(force=True, silent=True) or {}
    side_a = payload.get("side_a", [])
    side_b = payload.get("side_b", [])
    num_games = int(payload.get("num_games", 20))

    a_totals = _sum_side([n for n in side_a if n], num_games)
    b_totals = _sum_side([n for n in side_b if n], num_games)
    a_score  = _overall_score(a_totals)
    b_score  = _overall_score(b_totals)
    diff     = round(a_score - b_score, 3)
    verdict  = "Side A looks better" if diff > 0 else ("Side B looks better" if diff < 0 else "Even trade")

    return jsonify({
        "z_categories": Z_CATS,
        "a_totals": a_totals,
        "b_totals": b_totals,
        "a_score": a_score,
        "b_score": b_score,
        "diff": diff,
        "verdict": verdict
    })
