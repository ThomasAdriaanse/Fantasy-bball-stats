# app/services/player_stats.py
import pandas as pd
from nba_api.stats.static import players as nba_players
from .s3_service import load_player_dataset_from_s3


def get_active_players_list():
    """
    Returns the same active players list you used before (dicts with 'full_name'),
    sorted alphabetically by full_name. Your template expects this shape.
    """
    active = nba_players.get_active_players()
    active.sort(key=lambda x: x.get("full_name", ""))
    return active


def build_chart_data(player_name: str, num_games: int, stat: str = "FPTS"):
    """
    Load player's full dataset from S3 and return compact chart data
    for the selected stat as a Python list (no files written).

    Output rows (same keys as before):
        {
          "Game_Number": int,
          "Centered_Avg_Stat": float,
          "MATCHUP": str | None,
          "SEASON": str | None,
          "SEASON_ID": str | None,
          "STAT": str  # the stat actually used
        }
    """
    df = load_player_dataset_from_s3(player_name)
    if df is None or df.empty:
        print(f"[S3] No data for {player_name}")
        return []

    # Ensure meta columns exist so template/Jinja never breaks
    required_meta = ["Game_Number", "MATCHUP", "SEASON", "SEASON_ID"]
    for col in required_meta:
        if col not in df.columns:
            df[col] = None

    # Allowed columns exactly as before
    base_allowed = {
        "FPTS",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "MIN",
        "PLUS_MINUS",
    }
    z_allowed = {"AVG_Z"} | {
        f"Z_{k}"
        for k in [
            "PTS",
            "FG3M",
            "REB",
            "AST",
            "STL",
            "BLK",
            "FG_PCT",
            "FT_PCT",
            "TOV",
        ]
    }

    value_col = stat if stat in (base_allowed | z_allowed) else "FPTS"
    if value_col not in df.columns:
        print(f"[WARN] '{value_col}' missing for {player_name}; falling back to FPTS.")
        value_col = "FPTS"
        if value_col not in df.columns:
            return []

    # Order by game number
    df = df.sort_values("Game_Number").reset_index(drop=True)

    # Centered moving average over ±num_games/2 like before
    # (Your previous code used a "half_window = num_games" and symmetric slice.)
    def centered_average(idx, half_window):
        start = max(0, idx - half_window)
        end = min(len(df), idx + half_window + 1)
        return pd.to_numeric(df[value_col].iloc[start:end], errors="coerce").mean()

    half_window = num_games
    df["Centered_Avg"] = df.index.map(lambda i: centered_average(i, half_window))

    chart_df = df[["Game_Number", "MATCHUP", "SEASON", "SEASON_ID"]].copy()
    chart_df["Centered_Avg_Stat"] = pd.to_numeric(df["Centered_Avg"], errors="coerce").round(3)
    chart_df["STAT"] = value_col
    chart_df = chart_df.dropna(subset=["Centered_Avg_Stat"])

    return chart_df[
        ["Game_Number", "Centered_Avg_Stat", "MATCHUP", "SEASON", "SEASON_ID", "STAT"]
    ].to_dict(orient="records")
