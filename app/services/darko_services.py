
import os
import csv
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Any

from app.services.z_score_calculations import raw_to_zscore
from app.services.s3_service import _safe_filename

# Configuration from env vars (consistent with docker-compose/sync)
DARKO_CACHE_DIR = os.getenv("DARKO_CACHE_DIR", "/app/data/player_darko")
TEAM_PACE_CACHE_DIR = os.getenv("TEAM_PACE_CACHE_DIR", "/app/data/team_pace")
SEASON_AVGS_CACHE_DIR = os.getenv("SEASON_AVGS_CACHE_DIR", "/app/data/season_avgs")

def _load_team_pace() -> Dict[str, float]:
    """
    Loads team pace data from JSON.
    Returns dict: { "Team Name": PaceValue, ... }
    """
    path = Path(TEAM_PACE_CACHE_DIR) / "team_pace.json"
    if not path.exists():
        print(f"[DARKO] Team pace file not found: {path}")
        return {}
    
    try:
        with path.open("r") as f:
            data = json.load(f)
        
        # Structure is: { "data": [ { "TEAM_NAME": "Miami Heat", "PACE": 105.5 }, ... ] }
        rows = data.get("data", [])
        pace_map = {}
        for row in rows:
            name = row.get("TEAM_NAME")
            pace = row.get("PACE")
            if name and pace:
                pace_map[name] = float(pace)
        return pace_map
    except Exception as e:
        print(f"[DARKO] Error loading team pace: {e}")
        return {}

def _load_season_averages() -> Dict[str, Dict[str, float]]:
    """
    Loads pre-calculated season averages (MPG etc.) from JSON.
    Returns dict: { "player_slug": { "MIN": 32.5, ... }, ... }
    """
    path = Path(SEASON_AVGS_CACHE_DIR) / "season_averages.json"
    if not path.exists():
        print(f"[DARKO] Season averages file not found: {path}")
        return {}
        
    try:
        with path.open("r") as f:
            payload = json.load(f)
        return payload.get("data", {})
    except Exception as e:
        print(f"[DARKO] Error loading season averages: {e}")
        return {}

def get_raw_darko_stats() -> List[Dict[str, Any]]:
    """
    Returns a list of players with their raw computed per-game DARKO stats.
    Logic:
      1. Load DARKO CSV (Per 100 Possessions)
      2. Load Team Pace
      3. Load Player Season Average (for MPG)
      4. Convert Per-100 to Per-Game:
         Stat_Per_Game = (Stat_100 / 100) * (TeamPace / 48) * MPG
    """
    # 1. Find DARKO CSV
    csv_files = glob.glob(os.path.join(DARKO_CACHE_DIR, "DARKO_player_talent_*.csv"))
    if not csv_files:
        print(f"[DARKO] No DARKO CSV found in {DARKO_CACHE_DIR}")
        return []
    
    # Use the first one found (sync logic ensures only one should be there usually, or we pick one)
    darko_path = csv_files[0]
    
    # 2. Load lookups
    pace_map = _load_team_pace()
    avgs_map = _load_season_averages()
    
    results = []
    
    try:
        with open(darko_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                player_name = row.get("Player", "").strip()
                team_name = row.get("Team", "").strip()
                
                if not player_name:
                    continue
                    
                # Lookup Pace
                # print log if missing
                if team_name not in pace_map:
                    print(f"[DARKO] Missing team pace for {team_name}")
                team_pace = pace_map.get(team_name, 100.0) # Default to 100 if missing? Or skip?
                
                # Lookup MPG
                # print log if missing
                if player_name not in avgs_map:
                    print(f"[DARKO] Missing player MPG for {player_name}")
                slug = _safe_filename(player_name)
                player_avg = avgs_map.get(slug, {})
                mpg = player_avg.get("MIN", 0.0)

                # Formula: (Val_Per_100 / 100) * ( (Team_Pace * MPG) / 48 ) ?
                # Pace = Possessions per 48 minutes.
                # Possessions per minute = Pace / 48.
                # Total Possessions for player = (Pace / 48) * MPG.
                # Stat = (Stat_Per_100 / 100) * Total_Possessions
                #      = (Stat_Per_100 / 100) * (Pace / 48 * MPG)
                #      = Stat_Per_100 * (Pace * MPG) / 4800
                
                conversion_factor = (team_pace * mpg) / 4800.0
                

                # FGA/100, FG2%, FG3A/100, FG3%, FG3ARate%, RimFGA/100, RimFG%, FTA/100, FT%, FTARate%, USG%, REB/100, AST/100, AST%, BLK/100, BLK%, STL/100, STL%, TOV/100

                
                def get_float(k):
                    try:
                        return float(row.get(k, 0))
                    except:
                        return 0.0

                fga_100 = get_float("FGA/100")
                fg3a_100 = get_float("FG3A/100")
                fg3_pct = get_float("FG3%")
                fta_100 = get_float("FTA/100")
                ft_pct = get_float("FT%")
                reb_100 = get_float("REB/100")
                ast_100 = get_float("AST/100")
                blk_100 = get_float("BLK/100")
                stl_100 = get_float("STL/100")
                tov_100 = get_float("TOV/100")
                
                # Derived 100 stats
                fg3m_100 = fg3a_100 * fg3_pct
                ftm_100 = fta_100 * ft_pct
                
                # FG2A = FGA - FG3A.
                # FG2M = FG2A * FG2_Pct.
                # Total FGM = FG2M + FG3M.
                
                fg2_pct = get_float("FG2%")
                fg2a_100 = fga_100 - fg3a_100
                fg2m_100 = fg2a_100 * fg2_pct
                fgm_100 = fg2m_100 + fg3m_100
                
                # deriving points from other stats
                # PTS = (FG2M * 2) + (FG3M * 3) + (FTM * 1)
                pts_100 = (fg2m_100 * 2) + (fg3m_100 * 3) + (ftm_100 * 1)
                
                # Convert to Per Game
                stats = {
                    "PTS": pts_100 * conversion_factor,
                    "REB": reb_100 * conversion_factor,
                    "AST": ast_100 * conversion_factor,
                    "STL": stl_100 * conversion_factor,
                    "BLK": blk_100 * conversion_factor,
                    "TOV": tov_100 * conversion_factor,
                    "FG3M": fg3m_100 * conversion_factor,
                    "FGM": fgm_100 * conversion_factor,
                    "FGA": fga_100 * conversion_factor,
                    "FTM": ftm_100 * conversion_factor,
                    "FTA": fta_100 * conversion_factor,
                }
                
                # Add Metadata
                stats["player_name"] = player_name
                stats["team"] = team_name
                stats["mpg"] = mpg
                stats["pace"] = team_pace
                
                results.append(stats)
                
    except Exception as e:
        print(f"[DARKO] Error processing CSV: {e}")
        return []
        
    return results

def get_darko_z_scores() -> List[Dict[str, Any]]:
    """
    Returns list of players with Z-Scores calculated from DARKO projections,
    AND "Real" Z-Scores calculated from season averages.
    """
    # 1. Get DARKO raw stats (which already loads season avgs internally, but we need them again)
    # Refactor idea: optimize later. For now just load again to be safe/simple.
    season_avgs = _load_season_averages()
    darko_raw_data = get_raw_darko_stats()
    
    results = []
    
    for darko_player in darko_raw_data:
        # 1. DARKO Z-Scores
        z_darko = raw_to_zscore(darko_player)
        
        # 2. Real Z-Scores
        # get_raw_darko_stats uses _safe_filename(player_name) to look up MPG.
        # We can do the same to find the season avg object.
        player_name = darko_player.get("player_name")
        slug = _safe_filename(player_name)
        real_stats = season_avgs.get(slug, {})
        
        # raw_to_zscore needs: PTS, FG3M, REB, AST, STL, BLK, TOV, FGM, FGA, FTM, FTA 
        
        if real_stats:
            z_real = raw_to_zscore(real_stats)
        else:
            # Empty Z-scores if no real data found
            z_real = {}
            # log error
            print(f"[DARKO] No real stats found for {player_name}")

        # Combine info
        combined = {
            "player_name": player_name,
            "team": darko_player.get("team"),
            "RAW_DARKO": {k: v for k,v in darko_player.items() if k not in ["player_name", "team"]},
            "RAW_REAL": real_stats,
            "Z_DARKO": z_darko,
            "Z_REAL": z_real
        }
        results.append(combined)
        
    return results
