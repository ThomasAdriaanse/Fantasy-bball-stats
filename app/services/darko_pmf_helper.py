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
    
    #print(f"[DARKO-PMF] Loaded {len(lookup)} players from DARKO")
    #print(f"[DARKO-PMF] Sample names: {list(lookup.keys())[:10]}")
    #print(f"[DARKO-PMF] Names with 'wash': {[n for n in lookup.keys() if 'wash' in n.lower()]}")
    
    return lookup


def _shift_pmf_1d(pmf: PMF1D, shift_amount: float) -> PMF1D:
    """
    Shift a 1D PMF by a float amount.
    distributes mass to neighboring integers to preserve the mean shift exacty.
    
    If shift is 1.5:
      k=1, eps=0.5
      p[i] goes to:
        i+1 with mass 0.5
        i+2 with mass 0.5
    """
    if abs(shift_amount) < 1e-9:
        return pmf.copy()

    old_p = pmf.p
    n = len(old_p)
    
    # Decompose shift
    # shift = k + epsilon
    # where k is integer, 0 <= epsilon < 1
    # Example: shift = 1.5 -> k=1, eps=0.5
    # Example: shift = -1.5 -> k=-2, eps=0.5  (-2 + 0.5 = -1.5)
    
    k = int(np.floor(shift_amount))
    epsilon = shift_amount - k
    
    # We will accumulate into a new array. 
    # The max index could shift right by (n + k + 1). Use a safe buffer then trim?
    # Or just iterate. Vectorized is better.
    
    # new_p initialized to zeroes
    # We need a size that covers the shift. 
    # But PMF1D usually expects to start at 0.
    # If we shift left, we might clip at 0. Logic needs to handle clipping or just wrap?
    # Usually stats are non-negative. If we shift left past 0, generally we stack at 0.
    
    # Let's create a target array of same size N? 
    # Or should we expand? If a player improves, they might exceed their old max score.
    # Usually PMFs auto-expand in convolution, but here we are mutating the base.
    # Let's expand if shifting right.
    
    new_len = n + abs(k) + 2 # generous buffer
    new_p = np.zeros(new_len, dtype=float)
    
    indices = np.arange(n)
    
    # Target indices
    idx_k = indices + k
    idx_k_plus_1 = indices + k + 1
    
    # Mass parts
    mass_k = old_p * (1.0 - epsilon)
    mass_k_plus_1 = old_p * epsilon
    
    # Add to new_p
    # mask for valid > 0 (assuming we don't handle negative stats)
    
    # We use np.add.at because multiple old indices might map to same new index 
    # (though with constant shift they map 1-to-1, but clipping might overlap)
    
    # Handle idx_k
    valid_mask = (idx_k >= 0) & (idx_k < new_len)
    np.add.at(new_p, idx_k[valid_mask], mass_k[valid_mask])
    
    # Clip logic: if idx < 0, dump into 0?
    # For fantasy stats, negative stats (except maybe +/-) are impossible.
    # If we shift a bad game even lower, it stays 0.
    clip_mask = (idx_k < 0)
    if clip_mask.any():
        new_p[0] += mass_k[clip_mask].sum()
        
    # Handle idx_k_plus_1
    valid_mask_2 = (idx_k_plus_1 >= 0) & (idx_k_plus_1 < new_len)
    np.add.at(new_p, idx_k_plus_1[valid_mask_2], mass_k_plus_1[valid_mask_2])
    
    clip_mask_2 = (idx_k_plus_1 < 0)
    if clip_mask_2.any():
        new_p[0] += mass_k_plus_1[clip_mask_2].sum()
        
    # Trim trailing zeros
    # normalize just in case
    s = new_p.sum()
    if s > 0:
        new_p /= s
        
    return PMF1D(new_p)


def _scale_pmf_2d(pmf2d: PMF2D, makes_scale: float, attempts_scale: float) -> PMF2D:
    """
    Scale a 2D PMF's axes by given factors using bilinear mass distribution.
    This prevents aliasing and gaps by distributing probability from (m,a) 
    to the 4 integer neighbors of the target (m*scale, a*scale).
    """
    # If scales are close to 1.0, return copy or skip
    if abs(makes_scale - 1.0) < 0.05 and abs(attempts_scale - 1.0) < 0.05:
        return pmf2d.copy()
    
    # Clamp scales
    makes_scale = np.clip(makes_scale, 0.5, 2.0)
    attempts_scale = np.clip(attempts_scale, 0.5, 2.0)
    
    old_grid = pmf2d.p
    old_m_size, old_a_size = old_grid.shape
    
    # Determine new size
    # Usually max index scales by approx factor.
    new_m_size = int(np.ceil((old_m_size - 1) * makes_scale)) + 2
    new_a_size = int(np.ceil((old_a_size - 1) * attempts_scale)) + 2
    
    new_grid = np.zeros((new_m_size, new_a_size), dtype=float)
    
    # Get indices of non-zero probabilities
    m_indices, a_indices = np.nonzero(old_grid)
    if len(m_indices) == 0:
        return PMF2D(np.zeros((1,1)))
        
    probs = old_grid[m_indices, a_indices]
    
    # Calculate target coordinates
    target_m = m_indices * makes_scale
    target_a = a_indices * attempts_scale
    
    # Integer floors and fractional parts
    m_floor = np.floor(target_m).astype(int)
    a_floor = np.floor(target_a).astype(int)
    
    delta_m = target_m - m_floor
    delta_a = target_a - a_floor
    
    # 4 neighbors
    # (m, a)     weight: (1-dm)*(1-da)
    # (m+1, a)   weight: dm * (1-da)
    # (m, a+1)   weight: (1-dm) * da
    # (m+1, a+1) weight: dm * da
    
    w00 = (1.0 - delta_m) * (1.0 - delta_a)
    w10 = delta_m * (1.0 - delta_a)
    w01 = (1.0 - delta_m) * delta_a
    w11 = delta_m * delta_a
    
    # Add to grid
    # We must clip indices just in case, but sizing above should suffice.
    # Actually, let's clamp to new_m_size-1
    
    def safe_add(m_idx, a_idx, mass):
        # Clip to valid bounds
        np.clip(m_idx, 0, new_m_size - 1, out=m_idx)
        np.clip(a_idx, 0, new_a_size - 1, out=a_idx)
        # Use simple flattening or just iterate? 
        # add.at works with tuple indices for 2D
        np.add.at(new_grid, (m_idx, a_idx), mass)

    safe_add(m_floor,     a_floor,     probs * w00)
    safe_add(m_floor + 1, a_floor,     probs * w10)
    safe_add(m_floor,     a_floor + 1, probs * w01)
    safe_add(m_floor + 1, a_floor + 1, probs * w11)
    
    # Normalize
    total = new_grid.sum()
    if total > 0:
        new_grid /= total
        
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

        if stat_col == "STL":
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
        
        if stat_col == "STL" and player_name == "Cason Wallace":
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
                if stat_col == "STL" and player_name == "Cason Wallace":
                    print(f"[DARKO-PMF-1D] {player_name} {stat_col}: DARKO={darko_value:.2f}, PMF={pmf_mean:.2f}, Shift={shift_amount:.2f}")
                
                # Apply shift
                adjusted_pmf = _shift_pmf_1d(base_pmf, shift_amount)
                if stat_col == "STL" and player_name == "Cason Wallace":
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
