# player/routes.py
from flask import Blueprint, render_template, request, jsonify
from app.services.player_stats import build_chart_data, get_active_players_list
from app.services.PMF_utils import (
    load_player_pmfs,
    compress_pmf,
    compress_ratio_pmf_from_2d
)

bp = Blueprint("players", __name__)  # endpoint name = "players"

@bp.route("/stats", methods=["GET", "POST"])
def player_stats():
    # defaults
    selected_player = request.values.get("player_name", "Jamal Murray")

    try:
        num_games = int(request.values.get("num_games", 20))
    except (ValueError, TypeError):
        num_games = 20
    
    # DEFAULT TO AVG_Z
    selected_stat = request.values.get("stat", "AVG_Z")

    # Organize stats into three categories
    stat_options = {
        'raw': ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3M', 'FT_PCT'],
        'z_score': ['Z_PTS', 'Z_FG3M', 'Z_REB', 'Z_AST', 'Z_STL', 'Z_BLK', 'Z_FG', 'Z_FT', 'Z_TOV'],
        'other': ['AVG_Z', 'FGM', 'FGA', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'MIN', 'PLUS_MINUS']
    }

    try:
        chart_data = build_chart_data(selected_player, num_games, stat=selected_stat)
    except Exception as e:
        print(f"[ERROR] Failed to build chart data for {selected_player}: {e}")
        chart_data = []
    
    players = get_active_players_list()

    # --- Load PMF Data (Season Distribution) ---
    pmf_data = {}
    try:
        raw_pmfs = load_player_pmfs(selected_player)

        if raw_pmfs:
            # We only want specific categories
            cats = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG%", "FT%"]
            
            # Mapping from UI/target cat to PMF key if different
            # 1D keys: PTS, REB, AST, STL, BLK, FG3M, TOV
            # 2D keys: FG, FT
            
            key_map = {
                "PTS": "PTS", "REB": "REB", "AST": "AST",
                "STL": "STL", "BLK": "BLK", "3PM": "FG3M", "TO": "TOV",
                "FG%": "FG", "FT%": "FT"
            }

            # Check 1D
            p1d = raw_pmfs.get("1d", {})
            # Check 2D
            p2d = raw_pmfs.get("2d", {})

            for cat in cats:
                pmf_key = key_map.get(cat, cat)
                
                if cat in ["FG%", "FT%"]:
                    if pmf_key in p2d:
                        pmf_obj = p2d[pmf_key]
                        compressed = compress_ratio_pmf_from_2d(pmf_obj)
                        compressed["mean"] = pmf_obj.means()
                        pmf_data[cat] = compressed
                else:
                    if pmf_key in p1d:
                        pmf_obj = p1d[pmf_key]
                        compressed = compress_pmf(pmf_obj)
                        compressed["mean"] = pmf_obj.mean()
                        pmf_data[cat] = compressed
    except Exception as e:
        print(f"[ERROR] Failed to load PMF data for {selected_player}: {e}")
        pmf_data = {}

    return render_template(
        "player_stats.html",
        players=players,
        selected_player=selected_player,
        num_games=num_games,
        stat_options=stat_options,
        selected_stat=selected_stat,
        chart_data_json=chart_data,
        pmf_data=pmf_data
    )

@bp.get("/api/player_stats")
def api_player_stats():
    player_name = request.args.get("player_name", "Jamal Murray")
    num_games   = int(request.args.get("num_games", 20))
    # DEFAULT TO AVG_Z
    stat        = request.args.get("stat", "AVG_Z")
    data = build_chart_data(player_name, num_games, stat)
    return jsonify(data)

@bp.get("/data/player_stats.json")
def data_player_stats_json():
    player_name = request.args.get("player_name", "Jamal Murray")
    num_games   = int(request.args.get("num_games", 20))
    # DEFAULT TO AVG_Z
    stat        = request.args.get("stat", "AVG_Z")
    data = build_chart_data(player_name, num_games, stat)
    return jsonify(data)
