from flask import Blueprint, render_template, request, jsonify
from app.services.player_stats import build_chart_data, get_active_players_list

bp = Blueprint("players", __name__)  # endpoint name = "players"

@bp.route("/stats", methods=["GET", "POST"])
def player_stats():
    # defaults
    selected_player = request.values.get("player_name", "Bol Bol")
    num_games = int(request.values.get("num_games", 20))
    selected_stat = request.values.get("stat", "FPTS")

    stat_options = [
        'FPTS','PTS','REB','AST','STL','BLK','TOV',
        'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT',
        'FTM','FTA','FT_PCT','MIN','PLUS_MINUS',
        'AVG_Z','Z_PTS','Z_FG3M','Z_REB','Z_AST','Z_STL','Z_BLK','Z_FG_PCT','Z_FT_PCT','Z_TOV'
    ]

    chart_data = build_chart_data(selected_player, num_games, stat=selected_stat)
    players = get_active_players_list()

    return render_template(
        "player_stats.html",
        players=players,
        selected_player=selected_player,
        num_games=num_games,
        stat_options=stat_options,
        selected_stat=selected_stat,
        chart_data_json=chart_data  # dict/list is fine; Jinja will json.dumps it in template
    )

@bp.get("/api/player_stats")
def api_player_stats():
    player_name = request.args.get("player_name", "Bol Bol")
    num_games   = int(request.args.get("num_games", 20))
    stat        = request.args.get("stat", "FPTS")
    data = build_chart_data(player_name, num_games, stat)
    return jsonify(data)

@bp.get("/data/player_stats.json")
def data_player_stats_json():
    # optional compat endpoint, same payload
    player_name = request.args.get("player_name", "Bol Bol")
    num_games   = int(request.args.get("num_games", 20))
    stat        = request.args.get("stat", "FPTS")
    data = build_chart_data(player_name, num_games, stat)
    return jsonify(data)
