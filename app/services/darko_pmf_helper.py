# app/services/darko_pmf_helper.py
"""
DARKO-adjusted PMF building helpers.

This module provides functions to build team PMFs adjusted by DARKO projections
without modifying the existing PMF infrastructure.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from .PMF_utils import (
    load_player_pmfs,
    convolve_pmf_n_times,
    convolve_pmf_2d_n_times,
)
from .classes.pmf1d import PMF1D
from .classes.pmf2d import PMF2D
from . import darko_services


def _get_darko_lookup() -> Dict[str, Dict[str, Any]]:
    """
    Get DARKO data and create a player name lookup dictionary.
    Returns dict mapping player_name -> DARKO data.
    """
    darko_data = darko_services.get_darko_z_scores()
    lookup = {}
    for p in darko_data:
        name = p.get("player_name")
        if name:
            lookup[name] = p
    
    print(f"[DARKO-PMF] Loaded {len(lookup)} players from DARKO")
    print(f"[DARKO-PMF] Sample names: {list(lookup.keys())[:10]}")
    print(f"[DARKO-PMF] Names with 'wash': {[n for n in lookup.keys() if 'wash' in n.lower()]}")
    
    return lookup


def _shift_pmf_1d(pmf: PMF1D, shift_amount: float) -> PMF1D:
    """
    Shift a 1D PMF by a constant amount.
    
    This effectively moves the distribution left or right on the number line.
    For example, if a player's PMF has mean 20 PTS but DARKO projects 22 PTS,
    we shift by +2.
    
    Args:
        pmf: Original PMF
        shift_amount: Amount to shift (can be negative)
    
    Returns:
        New PMF with shifted distribution
    """
    if abs(shift_amount) < 0.01:  # No meaningful shift
        return pmf.copy()
    
    # Round shift to nearest integer for discrete PMF
    shift_int = int(round(shift_amount))
    
    if shift_int == 0:
        return pmf.copy()
    
    # Create new probability array
    old_probs = pmf.p
    old_len = len(old_probs)
    
    if shift_int > 0:
        # Shift right: add zeros at the beginning
        new_probs = np.concatenate([np.zeros(shift_int), old_probs])
    else:
        # Shift left: remove from beginning or add to end
        abs_shift = abs(shift_int)
        if abs_shift >= old_len:
            # Shift is larger than distribution, return delta at 0
            return PMF1D(np.array([1.0]))
        new_probs = old_probs[abs_shift:]
    
    # Normalize to ensure sum = 1.0
    total = new_probs.sum()
    if total > 0:
        new_probs = new_probs / total
    
    return PMF1D(new_probs)


def _scale_pmf_2d(pmf2d: PMF2D, makes_scale: float, attempts_scale: float) -> PMF2D:
    """
    Scale a 2D PMF's axes by given factors.
    
    For percentage stats (FG%, FT%), we scale both the makes and attempts axes
    to match DARKO projections while preserving the correlation structure.
    
    Args:
        pmf2d: Original 2D PMF over (makes, attempts)
        makes_scale: Factor to scale makes axis (darko_makes / pmf_mean_makes)
        attempts_scale: Factor to scale attempts axis
    
    Returns:
        New PMF2D with scaled axes
    """
    # If scales are very close to 1.0, no adjustment needed
    if abs(makes_scale - 1.0) < 0.05 and abs(attempts_scale - 1.0) < 0.05:
        return pmf2d.copy()
    
    # Clamp scales to reasonable range to avoid extreme distortions
    makes_scale = np.clip(makes_scale, 0.5, 2.0)
    attempts_scale = np.clip(attempts_scale, 0.5, 2.0)
    
    # Get the original grid
    old_grid = pmf2d.p
    old_m_size, old_a_size = old_grid.shape
    
    # Calculate new grid size
    new_m_size = int(np.ceil(old_m_size * makes_scale))
    new_a_size = int(np.ceil(old_a_size * attempts_scale))
    
    # Create new grid
    new_grid = np.zeros((new_m_size, new_a_size), dtype=float)
    
    # Map old probabilities to new grid positions
    for old_m in range(old_m_size):
        for old_a in range(old_a_size):
            prob = old_grid[old_m, old_a]
            if prob < 1e-10:
                continue
            
            # Scale indices
            new_m = int(round(old_m * makes_scale))
            new_a = int(round(old_a * attempts_scale))
            
            # Ensure within bounds
            if 0 <= new_m < new_m_size and 0 <= new_a < new_a_size:
                new_grid[new_m, new_a] += prob
    
    # Normalize
    total = new_grid.sum()
    if total > 0:
        new_grid = new_grid / total
    
    return PMF2D(new_grid)


def build_team_pmf_counting_with_darko(
    team_players: List[Dict[str, Any]],
    stat_col: str,
    season: str,
    darko_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    games_field: str = "games",
    injury_field: str = "inj",
    skip_injury_status: tuple = ("OUT", "INJURY_RESERVE", "IR", "SUSPENDED"),
    include_injured_players: bool = True,
) -> PMF1D:
    """
    Build team PMF for counting stat, adjusted by DARKO projections.
    
    For each player:
    1. Load base PMF from cache
    2. Get DARKO projection
    3. Shift PMF to match DARKO mean
    4. Convolve for N games
    5. Add to team PMF
    """
    if darko_lookup is None:
        darko_lookup = _get_darko_lookup()
    
    team_pmf = PMF1D(np.array([1.0]))  # delta at 0
    
    # Map stat_col to DARKO raw key
    stat_to_darko_key = {
        "PTS": "PTS",
        "FG3M": "FG3M",
        "REB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TOV",
    }
    
    darko_key = stat_to_darko_key.get(stat_col)
    if not darko_key:
        # Fallback to regular PMF if stat not supported
        from .PMF_utils import build_team_pmf_counting
        return build_team_pmf_counting(
            team_players,
            stat_col=stat_col,
            season=season,
            load_player_pmfs=load_player_pmfs,
            games_field=games_field,
            injury_field=injury_field,
            skip_injury_status=skip_injury_status,
            include_injured_players=include_injured_players,
        )
    
    for player in team_players:
        player_name = player.get("player_name")
        games_remaining = int(player.get(games_field, 0))
        injury_status = str(player.get(injury_field, "ACTIVE"))
        
        if not player_name:
            continue
        if games_remaining <= 0:
            continue
        if injury_status in skip_injury_status and not include_injured_players:
            continue
        
        print(f"[DARKO-PMF-1D] Processing {player_name} for {stat_col}, games={games_remaining}")
        
        # Load base PMF
        pmf_payload = load_player_pmfs(player_name)
        if not pmf_payload or not isinstance(pmf_payload, dict):
            print(f"[DARKO-PMF-1D] No PMF payload for {player_name}")
            continue
        
        pmf_1d_block = pmf_payload.get("1d")
        if not isinstance(pmf_1d_block, dict):
            print(f"[DARKO-PMF-1D] No 1D block for {player_name}")
            continue
        
        stat_entry = pmf_1d_block.get(stat_col)
        if stat_entry is None:
            print(f"[DARKO-PMF-1D] No {stat_col} entry for {player_name}")
            continue
        
        # Convert to PMF1D
        if isinstance(stat_entry, PMF1D):
            base_pmf = stat_entry
        elif isinstance(stat_entry, dict):
            probs = np.array(stat_entry.get("probs", []), dtype=float)
            if probs.size == 0:
                continue
            base_pmf = PMF1D(probs)
        else:
            probs = np.array(stat_entry, dtype=float)
            if probs.size == 0:
                continue
            base_pmf = PMF1D(probs)
        
        print(f"[DARKO-PMF-1D] Base PMF mean for {player_name} {stat_col}: {base_pmf.mean():.2f}")
        
        # Get DARKO adjustment
        darko_data = darko_lookup.get(player_name)
        if darko_data:
            raw_darko = darko_data.get("RAW_DARKO", {})
            darko_value = raw_darko.get(darko_key)
            
            if darko_value is not None:
                # Calculate shift amount
                pmf_mean = base_pmf.mean()
                shift_amount = float(darko_value) - pmf_mean
                
                print(f"[DARKO-PMF-1D] {player_name} {stat_col}: DARKO={darko_value:.2f}, PMF={pmf_mean:.2f}, Shift={shift_amount:.2f}")
                
                # Apply shift
                adjusted_pmf = _shift_pmf_1d(base_pmf, shift_amount)
                print(f"[DARKO-PMF-1D] Adjusted PMF mean: {adjusted_pmf.mean():.2f}")
            else:
                print(f"[DARKO-PMF-1D] No DARKO value for {player_name} stat {darko_key}")
                adjusted_pmf = base_pmf
        else:
            print(f"[DARKO-PMF-1D] No DARKO data found for player: '{player_name}'")
            # No DARKO data, use base PMF
            adjusted_pmf = base_pmf
        
        # Convolve for N games
        total_pmf = convolve_pmf_n_times(adjusted_pmf, games_remaining)
        
        # Add to team
        team_pmf = team_pmf.convolve(total_pmf)
    
    return team_pmf


def build_team_pmf_2d_with_darko(
    team_players: List[Dict[str, Any]],
    makes_col: str,
    attempts_col: str,
    season: str,
    darko_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    games_field: str = "games",
    injury_field: str = "inj",
    skip_injury_status: tuple = ("OUT", "INJURY_RESERVE", "IR", "SUSPENDED"),
    include_injured_players: bool = True,
) -> PMF2D:
    """
    Build team 2D PMF for percentage stat, adjusted by DARKO projections.
    
    For each player:
    1. Load base 2D PMF from cache
    2. Get DARKO projections for makes and attempts
    3. Scale PMF axes to match DARKO
    4. Convolve for N games
    5. Add to team PMF
    """
    if darko_lookup is None:
        darko_lookup = _get_darko_lookup()
    
    team_pmf = PMF2D(np.array([[1.0]]))  # delta at (0,0)
    
    # Determine which stat (FG or FT)
    if makes_col == "FGM" and attempts_col == "FGA":
        key_2d = "FG"
        darko_makes_key = "FGM"
        darko_attempts_key = "FGA"
    elif makes_col == "FTM" and attempts_col == "FTA":
        key_2d = "FT"
        darko_makes_key = "FTM"
        darko_attempts_key = "FTA"
    else:
        # Fallback to regular PMF
        from .PMF_utils import build_team_pmf_2d
        return build_team_pmf_2d(
            team_players,
            makes_col=makes_col,
            attempts_col=attempts_col,
            season=season,
            load_player_pmfs=load_player_pmfs,
            games_field=games_field,
            injury_field=injury_field,
            skip_injury_status=skip_injury_status,
            include_injured_players=include_injured_players,
        )
    
    for player in team_players:
        player_name = player.get("player_name")
        games_remaining = int(player.get(games_field, 0))
        injury_status = str(player.get(injury_field, "ACTIVE"))
        
        if not player_name:
            continue
        if games_remaining <= 0:
            continue
        if injury_status in skip_injury_status and not include_injured_players:
            continue
        
        # Load base PMF
        pmf_payload = load_player_pmfs(player_name)
        if not pmf_payload or not isinstance(pmf_payload, dict):
            continue
        
        pmf_2d_block = pmf_payload.get("2d")
        if not isinstance(pmf_2d_block, dict):
            continue
        
        if key_2d not in pmf_2d_block:
            continue
        
        entry = pmf_2d_block[key_2d]
        
        # Convert to PMF2D
        if isinstance(entry, PMF2D):
            base_pmf = entry
        else:
            continue
        
        # Get DARKO adjustment
        darko_data = darko_lookup.get(player_name)
        if darko_data:
            raw_darko = darko_data.get("RAW_DARKO", {})
            darko_makes = raw_darko.get(darko_makes_key)
            darko_attempts = raw_darko.get(darko_attempts_key)
            
            if darko_makes is not None and darko_attempts is not None:
                # Calculate current PMF means directly
                arr = base_pmf.p
                m_idx, a_idx = np.indices(arr.shape)
                pmf_mean_makes = float((m_idx * arr).sum())
                pmf_mean_attempts = float((a_idx * arr).sum())
                
                # Calculate scale factors
                if pmf_mean_makes > 0 and pmf_mean_attempts > 0:
                    makes_scale = float(darko_makes) / pmf_mean_makes
                    attempts_scale = float(darko_attempts) / pmf_mean_attempts
                    
                    # Apply scaling
                    adjusted_pmf = _scale_pmf_2d(base_pmf, makes_scale, attempts_scale)
                else:
                    adjusted_pmf = base_pmf
            else:
                adjusted_pmf = base_pmf
        else:
            # No DARKO data, use base PMF
            adjusted_pmf = base_pmf
        
        # Convolve for N games
        total_pmf = convolve_pmf_2d_n_times(adjusted_pmf, games_remaining)
        
        # Add to team
        team_pmf = team_pmf.convolve(total_pmf)
        team_pmf.p[team_pmf.p < 1e-7] = 0.0  # trim for speed
    
    return team_pmf
