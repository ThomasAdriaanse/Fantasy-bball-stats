# app/services/compare_presenter.py
"""
Compare presenter for fantasy basketball matchups.
Uses histogram-based Monte Carlo simulation for win probability calculations.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import signal
from .s3_service import load_player_dataset_from_s3
import time


# Mapping from display categories to S3 data columns
CATEGORY_COLUMN_MAP = {
    'PTS': 'PTS',
    'REB': 'REB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    '3PM': 'FG3M',
    'TO': 'TOV',
}

# Categories that use percentage calculations
PERCENTAGE_CATEGORIES = {
    'FG%': ('FGM', 'FGA'),
    'FT%': ('FTM', 'FTA'),
}

CURRENT_SEASON = "2025-26"


def build_pmf_from_games(stat_values: np.ndarray) -> np.ndarray:
    """
    Build a probability mass function (PMF) from a player's game log.
    
    Args:
        stat_values: Array of stat values from actual games
    
    Returns:
        PMF array where pmf[k] = P(stat = k)
    """
    if len(stat_values) == 0:
        return np.array([1.0])  # Delta at 0
    
    # Round to integers for discrete distribution
    stat_values = np.round(stat_values).astype(int)
    
    # Handle negative values (shouldn't happen for counting stats, but just in case)
    min_val = max(0, stat_values.min())
    max_val = stat_values.max()
    
    # Build histogram
    counts = np.bincount(stat_values - min_val, minlength=max_val - min_val + 1)
    pmf = counts / counts.sum()
    
    return pmf


def convolve_pmf_n_times(pmf: np.ndarray, n: int) -> np.ndarray:
    """
    Convolve a PMF with itself n times (for n games) using FFT for speed.
    
    Args:
        pmf: Single-game PMF
        n: Number of games
    
    Returns:
        PMF for total over n games
    """
    if n == 0:
        return np.array([1.0])  # Delta at 0
    
    if n == 1:
        return pmf.copy()
    
    # Use FFT-based convolution for speed (scipy's fftconvolve is optimized)
    result = pmf.copy()
    for _ in range(n - 1):
        result = signal.fftconvolve(result, pmf, mode='full')
        # Renormalize to maintain probability distribution
        result = result / result.sum()
    
    return result


def calculate_team_pmf(
    team_players: List[Dict[str, Any]],
    category: str,
    season: str = CURRENT_SEASON
) -> np.ndarray:
    """
    Calculate the PMF for a team's total in a category using convolution.
    
    Args:
        team_players: List of player dicts with 'player_name' and 'games' (remaining)
        category: Category to calculate (e.g., 'PTS', 'REB')
        season: Season string for filtering data
    
    Returns:
        PMF array where pmf[k] = P(team total = k)
    """
    start_time = time.time()
    
    # Get the column name for this category
    stat_col = CATEGORY_COLUMN_MAP.get(category)
    if not stat_col:
        return np.array([1.0])  # Delta at 0
    
    # Start with delta at 0 (no contribution)
    team_pmf = np.array([1.0])
    
    print(f"  [PMF] Processing {len(team_players)} players for {category}")
    player_count = 0
    
    for player in team_players:
        player_name = player.get('player_name')
        games_remaining = player.get('games', 0)
        
        if not player_name or games_remaining <= 0:
            continue
        
        # Skip injured players
        injury_status = player.get('inj', 'ACTIVE')
        if injury_status in ('OUT', 'INJURY_RESERVE', 'IR', 'SUSPENDED'):
            print(f"  [PMF] Skipping {player_name} (injured: {injury_status})")
            continue
        
        player_count += 1
        player_start = time.time()
        print(f"  [PMF] [{player_count}] {player_name} ({games_remaining} games)")
        
        # Load player's game log from S3
        load_start = time.time()
        df = load_player_dataset_from_s3(player_name)
        load_time = time.time() - load_start
        
        if df is None or df.empty:
            print(f"  [PMF] ⚠️  No S3 data ({load_time:.2f}s)")
            continue
        
        # Filter to recent seasons (current and next season for rookies/summer league)
        # This allows both "2024-25" and "2025-26" data
        if 'SEASON' in df.columns:
            # Keep current season and next season (for rookies)
            current_year = int(season.split('-')[0])
            df = df[df['SEASON'].str.startswith(str(current_year)) | 
                    df['SEASON'].str.startswith(str(current_year + 1))]
        
        if df.empty or stat_col not in df.columns:
            print(f"  [PMF] ⚠️  No {stat_col} data")
            continue
        
        # Get valid stat values
        stat_values = df[stat_col].dropna().values
        if len(stat_values) == 0:
            print(f"  [PMF] ⚠️  No valid {stat_col} values")
            continue
        
        # Build single-game PMF for this player
        pmf_start = time.time()
        player_single_game_pmf = build_pmf_from_games(stat_values)
        pmf_time = time.time() - pmf_start
        
        # Convolve for multiple games
        conv_start = time.time()
        player_total_pmf = convolve_pmf_n_times(player_single_game_pmf, int(games_remaining))
        conv_time = time.time() - conv_start
        
        # Add this player's contribution to team total
        combine_start = time.time()
        team_pmf = signal.fftconvolve(team_pmf, player_total_pmf, mode='full')
        combine_time = time.time() - combine_start
        
        player_total = time.time() - player_start
        print(f"  [PMF]   ✓ {player_total:.2f}s (load:{load_time:.2f}s pmf:{pmf_time:.3f}s conv:{conv_time:.3f}s combine:{combine_time:.3f}s) PMF size: {len(team_pmf)}")
    
    total_time = time.time() - start_time
    print(f"  [PMF] ✓ Completed {category} in {total_time:.2f}s - Final PMF size: {len(team_pmf)}")
    return team_pmf


def calculate_win_probability(team1_pmf: np.ndarray, team2_pmf: np.ndarray) -> float:
    """
    Calculate P(Team 1 > Team 2) given their PMFs.
    
    Args:
        team1_pmf: PMF for team 1's total
        team2_pmf: PMF for team 2's total
    
    Returns:
        Probability that team 1 wins (0.0 to 1.0)
    """
    # Build CDF for team 2
    cdf_team2 = np.cumsum(team2_pmf)
    
    # Pad with leading 0 so cdf_shifted[k] = P(Team2 <= k-1) = P(Team2 < k)
    cdf_shifted = np.concatenate([[0.0], cdf_team2[:-1]])
    
    # Ensure both arrays are same length
    max_len = max(len(team1_pmf), len(cdf_shifted))
    team1_padded = np.pad(team1_pmf, (0, max_len - len(team1_pmf)))
    cdf_padded = np.pad(cdf_shifted, (0, max_len - len(cdf_shifted)))
    
    # P(Team1 > Team2) = sum of P(Team1 = k) * P(Team2 < k)
    win_prob = float(np.sum(team1_padded * cdf_padded))
    
    return win_prob


def build_snapshot_rows(
    team1_current_stats: Optional[Dict[str, Any]],
    team2_current_stats: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build snapshot comparison rows showing current stats.
    
    Displays current accumulated stats for each team with visual highlighting
    for the team that's winning each category.
    
    Returns list of dicts with structure:
    {
        "cat": str,           # Category name (PTS, REB, etc.)
        "v1": float,          # Team 1 value
        "v2": float,          # Team 2 value
        "disp1": str,         # Team 1 display value
        "disp2": str,         # Team 2 display value
        "a1": float,          # Team 1 alpha (highlight intensity 0-1)
        "a2": float,          # Team 2 alpha (highlight intensity 0-1)
    }
    """
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%', 'TO']
    
    # Scale factors for calculating highlight intensity
    # Larger differences relative to these scales = stronger highlight
    scale_map = {
        'PTS': 100,
        'REB': 30,
        'AST': 20,
        'STL': 15,
        'BLK': 10,
        '3PM': 15,
        'FG%': 0.50,  # 5 percentage points
        'FT%': 0.50,  # 5 percentage points
        'TO': 10,
    }
    
    rows = []
    t1 = team1_current_stats or {}
    t2 = team2_current_stats or {}
    
    for cat in categories:
        # Extract values from nested dict structure: {"PTS": {"value": 370.0, "result": "LOSS"}}
        stat1 = t1.get(cat, {})
        stat2 = t2.get(cat, {})
        
        v1 = stat1.get("value") if isinstance(stat1, dict) else None
        v2 = stat2.get("value") if isinstance(stat2, dict) else None
        
        # Default display and alpha
        disp1 = "-"
        disp2 = "-"
        a1 = 0.0
        a2 = 0.0
        
        # If both values exist, calculate highlighting
        if v1 is not None and v2 is not None:
            try:
                v1_float = float(v1)
                v2_float = float(v2)
                
                # Calculate difference and intensity
                diff = abs(v1_float - v2_float)
                scale = scale_map.get(cat, 10.0)
                intensity = min(diff / scale, 1.0)  # Clamp to 0-1
                
                # Alpha range: 0.25 (small lead) to 0.80 (large lead)
                alpha = 0.25 + 0.55 * intensity
                
                # Only color the WINNING team, with intensity based on margin
                # (lower is better for TO)
                if cat == 'TO':
                    if v1_float < v2_float:
                        a1 = alpha  # Team 1 wins (blue)
                        a2 = 0.0    # Team 2 loses (no color)
                    elif v2_float < v1_float:
                        a1 = 0.0    # Team 1 loses (no color)
                        a2 = alpha  # Team 2 wins (red)
                    # else: tie, both stay 0.0
                else:
                    if v1_float > v2_float:
                        a1 = alpha  # Team 1 wins (blue)
                        a2 = 0.0    # Team 2 loses (no color)
                    elif v2_float > v1_float:
                        a1 = 0.0    # Team 1 loses (no color)
                        a2 = alpha  # Team 2 wins (red)
                    # else: tie, both stay 0.0
                
                # Format display values
                if cat in ('FG%', 'FT%'):
                    # Convert decimal to percentage: 0.422 -> "42.2%"
                    disp1 = f"{v1_float * 100:.1f}%"
                    disp2 = f"{v2_float * 100:.1f}%"
                else:
                    # Show as-is for counting stats
                    disp1 = str(v1)
                    disp2 = str(v2)
                    
            except (ValueError, TypeError):
                # If conversion fails, leave as defaults
                pass
        else:
            # If only one value exists, still show it
            if cat in ('FG%', 'FT%'):
                if v1 is not None:
                    try:
                        disp1 = f"{float(v1) * 100:.1f}%"
                    except:
                        pass
                if v2 is not None:
                    try:
                        disp2 = f"{float(v2) * 100:.1f}%"
                    except:
                        pass
        
        rows.append({
            "cat": cat,
            "v1": v1,
            "v2": v2,
            "disp1": disp1,
            "disp2": disp2,
            "a1": round(a1, 3),
            "a2": round(a2, 3),
        })
    
    return rows


def build_odds_rows(
    win1_list: List[Dict[str, Any]],
    combined_jsons: Dict[str, List[Dict[str, Any]]],
    expected_pct_map: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    team1_current_stats: Optional[Dict[str, Any]] = None,
    team2_current_stats: Optional[Dict[str, Any]] = None,
    data_team_players_1: Optional[List[Dict[str, Any]]] = None,
    data_team_players_2: Optional[List[Dict[str, Any]]] = None,
    exclude_day_to_day: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build odds rows showing win probabilities for each category using exact convolution.
    
    For each category:
    1. Build PMF for Team 1's projected total (from player game logs)
    2. Build PMF for Team 2's projected total (from player game logs)
    3. Add current totals to the PMFs
    4. Calculate exact P(Team 1 wins) by comparing distributions
    
    Returns list of dicts with structure:
    {
        "cat": str,           # Category name
        "p1": float,          # Team 1 win probability (0-100)
        "p2": float,          # Team 2 win probability (0-100)
        "class_name": str,    # CSS class: "winner-left", "winner-right", or "is-tie"
        "mid_t1": str,        # Team 1 projected total (display)
        "mid_t2": str,        # Team 2 projected total (display)
    }
    """
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%', 'TO']
    
    # Validate inputs
    if not (team1_current_stats and team2_current_stats and 
            data_team_players_1 and data_team_players_2):
        print("[WARN] Missing required data for build_odds_rows, returning dummy data")
        return [
            {
                "cat": cat,
                "p1": 50.0,
                "p2": 50.0,
                "class_name": "is-tie",
                "mid_t1": "-",
                "mid_t2": "-",
            }
            for cat in categories
        ]
    
    # Filter out day-to-day players if requested
    if exclude_day_to_day:
        data_team_players_1 = [p for p in data_team_players_1 if p.get('inj') != 'DAY_TO_DAY']
        data_team_players_2 = [p for p in data_team_players_2 if p.get('inj') != 'DAY_TO_DAY']
    
    rows = []
    
    for cat in categories:
        print(f"[INFO] Calculating odds for: {cat}")
        
        # Skip percentage categories for now (need special handling)
        if cat in ('FG%', 'FT%'):
            rows.append({
                "cat": cat,
                "p1": 50.0,
                "p2": 50.0,
                "class_name": "is-tie",
                "mid_t1": "-",
                "mid_t2": "-",
            })
            continue
        
        # Get current totals
        t1_stat = team1_current_stats.get(cat, {})
        t2_stat = team2_current_stats.get(cat, {})
        
        t1_current = t1_stat.get('value', 0.0) if isinstance(t1_stat, dict) else 0.0
        t2_current = t2_stat.get('value', 0.0) if isinstance(t2_stat, dict) else 0.0
        
        try:
            # Calculate PMFs for projected totals (remaining games only)
            print(f"  [ODDS] Calculating Team 1 PMF for {cat}...")
            t1_projected_pmf = calculate_team_pmf(data_team_players_1, cat)
            
            print(f"  [ODDS] Calculating Team 2 PMF for {cat}...")
            t2_projected_pmf = calculate_team_pmf(data_team_players_2, cat)
            
            # Add current totals by shifting the PMF
            # If current = 100 and PMF is for projected, then final PMF starts at index 100
            t1_current_int = int(round(t1_current))
            t2_current_int = int(round(t2_current))
            
            print(f"  [ODDS] Adding current totals (T1: {t1_current_int}, T2: {t2_current_int})")
            
            # Pad with zeros to shift the distribution
            t1_final_pmf = np.pad(t1_projected_pmf, (t1_current_int, 0))
            t2_final_pmf = np.pad(t2_projected_pmf, (t2_current_int, 0))
            
            print(f"  [ODDS] Calculating win probability...")
            
            # Calculate win probability
            # For TO (turnovers), lower is better, so flip the comparison
            if cat == 'TO':
                p_team1 = calculate_win_probability(t2_final_pmf, t1_final_pmf)  # Flipped
                p_team1 = 1.0 - p_team1  # Convert back to P(Team1 wins)
            else:
                p_team1 = calculate_win_probability(t1_final_pmf, t2_final_pmf)
            
            p_team1_pct = p_team1 * 100.0
            p_team2_pct = 100.0 - p_team1_pct
            
            print(f"  [ODDS] ✓ {cat}: Team1 {p_team1_pct:.1f}% vs Team2 {p_team2_pct:.1f}%")
            
            # Calculate expected values (mean of PMF)
            t1_indices = np.arange(len(t1_final_pmf))
            t2_indices = np.arange(len(t2_final_pmf))
            t1_expected = np.sum(t1_indices * t1_final_pmf)
            t2_expected = np.sum(t2_indices * t2_final_pmf)
            
            # Format display values
            mid_t1 = str(round(t1_expected))
            mid_t2 = str(round(t2_expected))
            
            # Determine CSS class
            if abs(p_team1_pct - p_team2_pct) < 0.5:
                class_name = "is-tie"
            elif p_team1_pct > p_team2_pct:
                class_name = "winner-left"
            else:
                class_name = "winner-right"
            
            rows.append({
                "cat": cat,
                "p1": round(p_team1_pct, 1),
                "p2": round(p_team2_pct, 1),
                "class_name": class_name,
                "mid_t1": mid_t1,
                "mid_t2": mid_t2,
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate {cat}: {e}")
            import traceback
            traceback.print_exc()
            # Return tie as fallback
            rows.append({
                "cat": cat,
                "p1": 50.0,
                "p2": 50.0,
                "class_name": "is-tie",
                "mid_t1": "-",
                "mid_t2": "-",
            })
    
    return rows
