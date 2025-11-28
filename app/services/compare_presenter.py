# app/services/compare_presenter.py
"""
Compare presenter for fantasy basketball matchups.
Uses histogram-based convolution for exact win probability calculations.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy import signal
from .s3_service import load_player_dataset_from_s3
import time

# Toggle debug prints
DEBUG_COMPARE_PRESENTER = True

# Mapping from display categories to S3 data columns (counting stats)
CATEGORY_COLUMN_MAP = {
    'PTS': 'PTS',
    'REB': 'REB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    '3PM': 'FG3M',
    'TO':  'TOV',
}

# Categories that use percentage calculations (makes, attempts)
PERCENTAGE_CATEGORIES = {
    'FG%': ('FGM', 'FGA'),
    'FT%': ('FTM', 'FTA'),
}

CURRENT_SEASON = "2025-26"

# How aggressively to trim tails:
TAIL_MASS_1D = 0.003
TAIL_MASS_RATIO = 0.003
MIN_BINS_AFTER_TRIM = 10  # don't over-trim extremely narrow distributions

# Number of buckets for % PMFs: 0.0% .. 100.0% in 0.1% steps
RATIO_BUCKETS = 1001  # indices 0..1000 -> 0.0..100.0


# ========= 1D PMF HELPERS (COUNTING STATS) =========

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

    result = pmf.copy()
    for _ in range(n - 1):
        result = signal.fftconvolve(result, pmf, mode='full')
        result = result / result.sum()
    return result


def trim_1d_pmf(pmf: np.ndarray,
                tail_mass: float = TAIL_MASS_1D,
                min_bins: int = MIN_BINS_AFTER_TRIM) -> np.ndarray:
    """
    Soft-trim extreme tails from a 1D PMF by zeroing out very small
    mass at both ends, then renormalizing.

    - Keeps the array length the same (indices still mean the same totals).
    - Uses a quantile-based cutoff so trimming adapts to the shape.
    """
    if pmf.size == 0:
        return pmf

    total = pmf.sum()
    if total <= 0:
        return pmf

    norm = pmf / total
    cdf = np.cumsum(norm)

    # Lower/upper quantile indices
    low_idx = int(np.searchsorted(cdf, tail_mass, side="left"))
    high_idx = int(np.searchsorted(cdf, 1.0 - tail_mass, side="right") - 1)

    # If trimming would leave too few bins, bail out
    if high_idx - low_idx + 1 < min_bins:
        return pmf

    trimmed = np.zeros_like(pmf)
    trimmed[low_idx:high_idx + 1] = pmf[low_idx:high_idx + 1]

    new_total = trimmed.sum()
    if new_total > 0:
        trimmed /= new_total
    else:
        # If everything got nuked, fallback to original
        return pmf

    if DEBUG_COMPARE_PRESENTER:
        print(
            f"[TRIM-1D] size={pmf.size}, "
            f"low_idx={low_idx}, high_idx={high_idx}, "
            f"tail_mass={tail_mass}"
        )

    return trimmed


def calculate_team_pmf(
    team_players: List[Dict[str, Any]],
    category: str,
    season: str = CURRENT_SEASON
) -> np.ndarray:
    """
    Calculate the PMF for a team's total in a counting category using convolution.
    """
    start_time = time.time()

    stat_col = CATEGORY_COLUMN_MAP.get(category)
    if not stat_col:
        return np.array([1.0])  # Delta at 0

    team_pmf = np.array([1.0])

    if DEBUG_COMPARE_PRESENTER:
        print(f"  [PMF] Processing {len(team_players)} players for {category}")
    player_count = 0

    for player in team_players:
        player_name = player.get('player_name')
        games_remaining = player.get('games', 0)

        if not player_name or games_remaining <= 0:
            continue

        injury_status = player.get('inj', 'ACTIVE')
        if injury_status in ('OUT', 'INJURY_RESERVE', 'IR', 'SUSPENDED'):
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF] Skipping {player_name} (injured: {injury_status})")
            continue

        player_count += 1
        player_start = time.time()
        if DEBUG_COMPARE_PRESENTER:
            print(f"  [PMF] [{player_count}] {player_name} ({games_remaining} games)")

        load_start = time.time()
        df = load_player_dataset_from_s3(player_name)
        load_time = time.time() - load_start

        if df is None or df.empty:
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF] ⚠️  No S3 data ({load_time:.2f}s)")
            continue

        if 'SEASON' in df.columns:
            current_year = int(season.split('-')[0])
            df = df[
                df['SEASON'].str.startswith(str(current_year)) |
                df['SEASON'].str.startswith(str(current_year + 1))
            ]

        if df.empty or stat_col not in df.columns:
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF] ⚠️  No {stat_col} data")
            continue

        stat_values = df[stat_col].dropna().values
        if len(stat_values) == 0:
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF] ⚠️  No valid {stat_col} values")
            continue

        pmf_start = time.time()
        player_single_game_pmf = build_pmf_from_games(stat_values)
        pmf_time = time.time() - pmf_start

        conv_start = time.time()
        player_total_pmf = convolve_pmf_n_times(player_single_game_pmf, int(games_remaining))
        conv_time = time.time() - conv_start

        combine_start = time.time()
        team_pmf = signal.fftconvolve(team_pmf, player_total_pmf, mode='full')
        combine_time = time.time() - combine_start

        player_total = time.time() - player_start
        if DEBUG_COMPARE_PRESENTER:
            print(
                f"  [PMF]   ✓ {player_total:.2f}s "
                f"(load:{load_time:.2f}s pmf:{pmf_time:.3f}s conv:{conv_time:.3f}s "
                f"combine:{combine_time:.3f}s) PMF size: {len(team_pmf)}"
            )

    total_time = time.time() - start_time
    if DEBUG_COMPARE_PRESENTER:
        print(f"  [PMF] ✓ Completed {category} in {total_time:.2f}s - Final PMF size: {len(team_pmf)}")
    return team_pmf


def calculate_win_probability(team1_pmf: np.ndarray, team2_pmf: np.ndarray) -> float:
    """
    Calculate P(Team 1 > Team 2) given their PMFs (counting stats).
    """
    cdf_team2 = np.cumsum(team2_pmf)
    cdf_shifted = np.concatenate([[0.0], cdf_team2[:-1]])

    max_len = max(len(team1_pmf), len(cdf_shifted))
    team1_padded = np.pad(team1_pmf, (0, max_len - len(team1_pmf)))
    cdf_padded = np.pad(cdf_shifted, (0, max_len - len(cdf_shifted)))

    win_prob = float(np.sum(team1_padded * cdf_padded))
    return win_prob


def compress_pmf(pmf: np.ndarray) -> Dict[str, Any]:
    """
    Compress a 1D PMF into {'min': int, 'probs': [floats]} for the frontend tooltip,
    after trimming tiny tails.
    """
    if pmf.size == 0:
        return {'min': 0, 'probs': []}

    # Trim tails but keep indices the same length for internal calculations.
    pmf_trimmed = trim_1d_pmf(pmf)

    indices = np.where(pmf_trimmed > 0)[0]
    if len(indices) == 0:
        return {'min': 0, 'probs': []}

    start_idx = int(indices[0])
    end_idx = int(indices[-1])
    probs = [round(float(p), 6) for p in pmf_trimmed[start_idx:end_idx + 1]]
    return {'min': start_idx, 'probs': probs}


# ========= 2D PMF HELPERS (FG% / FT%) =========

def build_2d_pmf_from_games(makes: np.ndarray, attempts: np.ndarray) -> np.ndarray:
    """Single-game 2D PMF over (makes, attempts)."""
    if len(makes) == 0 or len(attempts) == 0:
        pmf = np.zeros((1, 1), dtype=float)
        pmf[0, 0] = 1.0
        return pmf

    makes = np.round(makes).astype(int)
    attempts = np.round(attempts).astype(int)
    makes = np.clip(makes, 0, None)
    attempts = np.clip(attempts, 0, None)
    makes = np.minimum(makes, attempts)

    max_m = makes.max()
    max_a = attempts.max()
    pmf = np.zeros((max_m + 1, max_a + 1), dtype=float)
    for m, a in zip(makes, attempts):
        pmf[m, a] += 1.0

    total = pmf.sum()
    if total == 0:
        pmf[0, 0] = 1.0
    else:
        pmf /= total
    return pmf


def convolve_pmf_2d_n_times(pmf2d: np.ndarray, n: int) -> np.ndarray:
    """Convolve 2D PMF with itself n times (for n games)."""
    if n == 0:
        out = np.zeros((1, 1), dtype=float)
        out[0, 0] = 1.0
        return out
    if n == 1:
        return pmf2d.copy()

    result = pmf2d.copy()
    for _ in range(n - 1):
        result = signal.fftconvolve(result, pmf2d, mode='full')
        s = result.sum()
        if s > 0:
            result /= s
    return result


def calculate_team_pmf_2d(
    team_players: List[Dict[str, Any]],
    category: str,
    season: str = CURRENT_SEASON
) -> np.ndarray:
    """
    2D PMF for team totals in FG%/FT%:
    pmf[m,a] = P(team makes = m, attempts = a) for remaining games.
    """
    if category not in PERCENTAGE_CATEGORIES:
        pmf = np.zeros((1, 1), dtype=float)
        pmf[0, 0] = 1.0
        return pmf

    makes_col, attempts_col = PERCENTAGE_CATEGORIES[category]
    start_time = time.time()

    team_pmf = np.zeros((1, 1), dtype=float)
    team_pmf[0, 0] = 1.0

    if DEBUG_COMPARE_PRESENTER:
        print(f"  [PMF-2D] Processing {len(team_players)} players for {category}")
    player_count = 0

    for player in team_players:
        player_name = player.get('player_name')
        games_remaining = player.get('games', 0)
        if not player_name or games_remaining <= 0:
            continue

        injury_status = player.get('inj', 'ACTIVE')
        if injury_status in ('OUT', 'INJURY_RESERVE', 'IR', 'SUSPENDED'):
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF-2D] Skipping {player_name} (injured: {injury_status})")
            continue

        player_count += 1
        player_start = time.time()
        if DEBUG_COMPARE_PRESENTER:
            print(f"  [PMF-2D] [{player_count}] {player_name} ({games_remaining} games)")

        load_start = time.time()
        df = load_player_dataset_from_s3(player_name)
        load_time = time.time() - load_start

        if df is None or df.empty:
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF-2D] ⚠️  No S3 data ({load_time:.2f}s)")
            continue

        if 'SEASON' in df.columns:
            current_year = int(season.split('-')[0])
            df = df[
                df['SEASON'].str.startswith(str(current_year)) |
                df['SEASON'].str.startswith(str(current_year + 1))
            ]

        if df.empty or makes_col not in df.columns or attempts_col not in df.columns:
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF-2D] ⚠️  No {makes_col}/{attempts_col} data")
            continue

        makes = df[makes_col].dropna().values
        attempts = df[attempts_col].dropna().values
        if len(makes) == 0 or len(attempts) == 0:
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [PMF-2D] ⚠️  No valid {makes_col}/{attempts_col} values")
            continue

        pmf_start = time.time()
        player_single_2d = build_2d_pmf_from_games(makes, attempts)
        pmf_time = time.time() - pmf_start

        conv_start = time.time()
        player_total_2d = convolve_pmf_2d_n_times(player_single_2d, int(games_remaining))
        conv_time = time.time() - conv_start

        combine_start = time.time()
        team_pmf = signal.fftconvolve(team_pmf, player_total_2d, mode='full')
        s = team_pmf.sum()
        if s > 0:
            team_pmf /= s
        combine_time = time.time() - combine_start

        player_total = time.time() - player_start
        if DEBUG_COMPARE_PRESENTER:
            print(
                f"  [PMF-2D]   ✓ {player_total:.2f}s "
                f"(load:{load_time:.2f}s pmf:{pmf_time:.3f}s conv:{conv_time:.3f}s "
                f"combine:{combine_time:.3f}s) PMF shape: {team_pmf.shape}"
            )

    total_time = time.time() - start_time
    if DEBUG_COMPARE_PRESENTER:
        print(f"  [PMF-2D] ✓ Completed {category} in {total_time:.2f}s - Final PMF shape: {team_pmf.shape}")
    return team_pmf


def _ratios_and_probs_from_2d_pmf(pmf2d: np.ndarray):
    """Flatten 2D pmf[m,a] into (ratios, probs), ratio = m/a (0 if a=0)."""
    m_idx, a_idx = np.nonzero(pmf2d)
    if len(m_idx) == 0:
        return np.array([0.0]), np.array([1.0])

    probs = pmf2d[m_idx, a_idx]
    ratios = np.zeros_like(probs, dtype=float)
    mask = a_idx > 0
    ratios[mask] = m_idx[mask] / a_idx[mask]

    s = probs.sum()
    if s > 0:
        probs = probs / s
    return ratios, probs


def compress_ratio_pmf_from_2d(pmf2d: np.ndarray) -> Dict[str, Any]:
    """
    Compress a 2D (makes, attempts) PMF into a 1D PMF over percentage
    points with finer resolution, formatted as {'min': int, 'probs': [floats]}.

    We use RATIO_BUCKETS bins, where indices 0..(RATIO_BUCKETS-1) correspond
    to 0.0 .. 100.0% in equal steps (with RATIO_BUCKETS=1001 → 0.1% steps).
    Applies quantile-based tail trimming to avoid super-long graphs.
    """
    ratios, probs = _ratios_and_probs_from_2d_pmf(pmf2d)  # ratios in [0,1]

    if len(ratios) == 0:
        return {'min': 0, 'probs': []}

    # Bucket ratios into 0..100% with RATIO_BUCKETS bins.
    # With RATIO_BUCKETS = 1001, index k corresponds to k / 10.0 (%).
    buckets = np.zeros(RATIO_BUCKETS, dtype=float)
    perc_indices = np.clip(
        np.round(ratios * (RATIO_BUCKETS - 1)).astype(int),
        0,
        RATIO_BUCKETS - 1,
    )
    for idx, p in zip(perc_indices, probs):
        buckets[idx] += p

    total = buckets.sum()
    if total <= 0:
        return {'min': 0, 'probs': []}

    norm = buckets / total
    cdf = np.cumsum(norm)

    # Quantile-based trimming
    low_idx = int(np.searchsorted(cdf, TAIL_MASS_RATIO, side="left"))
    high_idx = int(np.searchsorted(cdf, 1.0 - TAIL_MASS_RATIO, side="right") - 1)

    if high_idx - low_idx + 1 < MIN_BINS_AFTER_TRIM:
        # Not enough bins to justify trimming; just keep region with >tiny mass
        indices = np.where(buckets > 1e-4)[0]
        if len(indices) == 0:
            return {'min': 0, 'probs': []}
        start = int(indices[0])
        end = int(indices[-1])
    else:
        start = low_idx
        end = high_idx

    probs_list = [round(float(p), 4) for p in norm[start:end + 1]]

    if DEBUG_COMPARE_PRESENTER:
        print(
            f"[TRIM-RATIO] buckets={len(buckets)}, start={start}, end={end}, "
            f"tail_mass={TAIL_MASS_RATIO}"
        )

    # NOTE: 'min' is in 0.1%-units when RATIO_BUCKETS=1001.
    # The frontend will convert to real % for plotting.
    return {'min': start, 'probs': probs_list}


def expected_ratio_from_2d_pmf(pmf2d: np.ndarray) -> float:
    """E[ratio] from 2D PMF."""
    r, p = _ratios_and_probs_from_2d_pmf(pmf2d)
    return float((r * p).sum())


def calculate_percentage_win_probability(
    team1_pmf2d: np.ndarray,
    team2_pmf2d: np.ndarray,
) -> float:
    """
    P(Team1 % > Team2 %) using ratio distributions from 2D PMFs.
    Ties counted as 0.5.
    """
    r1, p1 = _ratios_and_probs_from_2d_pmf(team1_pmf2d)
    r2, p2 = _ratios_and_probs_from_2d_pmf(team2_pmf2d)
    if len(r1) == 0 or len(r2) == 0:
        return 0.5

    order2 = np.argsort(r2)
    r2s = r2[order2]
    p2s = p2[order2]
    cdf2 = np.cumsum(p2s)

    idx_less = np.searchsorted(r2s, r1, side="left")
    idx_leq = np.searchsorted(r2s, r1, side="right")

    prob_less = np.where(idx_less > 0, cdf2[idx_less - 1], 0.0)
    prob_leq = np.where(idx_leq > 0, cdf2[idx_leq - 1], 0.0)

    win_states = p1 * prob_less
    tie_states = p1 * (prob_leq - prob_less)

    p_win = float(win_states.sum())
    p_tie = float(tie_states.sum())
    return p_win + 0.5 * p_tie


# ========= SNAPSHOT ROWS =========

def build_snapshot_rows(
    team1_current_stats: Optional[Dict[str, Any]],
    team2_current_stats: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build snapshot comparison rows showing current stats.
    """
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%', 'TO']

    scale_map = {
        'PTS': 100,
        'REB': 30,
        'AST': 20,
        'STL': 15,
        'BLK': 10,
        '3PM': 15,
        'FG%': 0.50,
        'FT%': 0.50,
        'TO': 10,
    }

    rows = []
    t1 = team1_current_stats or {}
    t2 = team2_current_stats or {}

    for cat in categories:
        stat1 = t1.get(cat, {})
        stat2 = t2.get(cat, {})

        v1 = stat1.get("value") if isinstance(stat1, dict) else None
        v2 = stat2.get("value") if isinstance(stat2, dict) else None

        disp1 = "-"
        disp2 = "-"
        a1 = 0.0
        a2 = 0.0

        if v1 is not None and v2 is not None:
            try:
                v1_float = float(v1)
                v2_float = float(v2)

                diff = abs(v1_float - v2_float)
                scale = scale_map.get(cat, 10.0)
                intensity = min(diff / scale, 1.0)
                alpha = 0.25 + 0.55 * intensity

                if cat == 'TO':
                    if v1_float < v2_float:
                        a1 = alpha
                        a2 = 0.0
                    elif v2_float < v1_float:
                        a1 = 0.0
                        a2 = alpha
                else:
                    if v1_float > v2_float:
                        a1 = alpha
                        a2 = 0.0
                    elif v2_float > v1_float:
                        a1 = 0.0
                        a2 = alpha

                if cat in ('FG%', 'FT%'):
                    disp1 = f"{v1_float * 100:.1f}%"
                    disp2 = f"{v2_float * 100:.1f}%"
                else:
                    disp1 = str(v1)
                    disp2 = str(v2)

            except (ValueError, TypeError):
                pass
        else:
            if cat in ('FG%', 'FT%'):
                if v1 is not None:
                    try:
                        disp1 = f"{float(v1) * 100:.1f}%"
                    except Exception:
                        pass
                if v2 is not None:
                    try:
                        disp2 = f"{float(v2) * 100:.1f}%"
                        pass
                    except Exception:
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


# ========= ODDS ROWS =========

def build_odds_rows(
    win1_list: List[Dict[str, Any]],
    combined_jsons: Dict[str, List[Dict[str, Any]]],
    expected_pct_map: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    team1_current_stats: Optional[Dict[str, Any]] = None,
    team2_current_stats: Optional[Dict[str, Any]] = None,
    data_team_players_1: Optional[List[Dict[str, Any]]] = None,
    data_team_players_2: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build odds rows showing win probabilities for each category using exact convolution.
    """
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%', 'TO']

    if not (team1_current_stats and team2_current_stats and
            data_team_players_1 and data_team_players_2):
        if DEBUG_COMPARE_PRESENTER:
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

    if DEBUG_COMPARE_PRESENTER:
        print("[DEBUG] team1_current_stats:", team1_current_stats)
        print("[DEBUG] team2_current_stats:", team2_current_stats)

    rows: List[Dict[str, Any]] = []

    for cat in categories:
        if DEBUG_COMPARE_PRESENTER:
            print(f"[INFO] Calculating odds for: {cat}")

        # ----- Percentage categories (FG%, FT%) using 2D PMFs -----
        if cat in ('FG%', 'FT%'):
            try:
                if cat == 'FG%':
                    t1_made = float(team1_current_stats.get('FGM', {}).get('value', 0.0))
                    t1_att = float(team1_current_stats.get('FGA', {}).get('value', 0.0))
                    t2_made = float(team2_current_stats.get('FGM', {}).get('value', 0.0))
                    t2_att = float(team2_current_stats.get('FGA', {}).get('value', 0.0))
                else:  # FT%
                    t1_made = float(team1_current_stats.get('FTM', {}).get('value', 0.0))
                    t1_att = float(team1_current_stats.get('FTA', {}).get('value', 0.0))
                    t2_made = float(team2_current_stats.get('FTM', {}).get('value', 0.0))
                    t2_att = float(team2_current_stats.get('FTA', {}).get('value', 0.0))

                if DEBUG_COMPARE_PRESENTER:
                    print(f"[FT%/FG% DEBUG] t1_cur=({t1_made},{t1_att}) t2_cur=({t2_made},{t2_att})")

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Calculating Team 1 2D PMF for {cat}...")
                t1_pmf_2d_proj = calculate_team_pmf_2d(data_team_players_1, cat)

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Calculating Team 2 2D PMF for {cat}...")
                t2_pmf_2d_proj = calculate_team_pmf_2d(data_team_players_2, cat)

                t1_made_i = int(round(t1_made))
                t1_att_i = int(round(t1_att))
                t2_made_i = int(round(t2_made))
                t2_att_i = int(round(t2_att))

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Adding current totals (T1: {t1_made_i}/{t1_att_i}, T2: {t2_made_i}/{t2_att_i})")

                t1_pmf_2d_final = np.pad(t1_pmf_2d_proj, ((t1_made_i, 0), (t1_att_i, 0)))
                t2_pmf_2d_final = np.pad(t2_pmf_2d_proj, ((t2_made_i, 0), (t2_att_i, 0)))

                s1 = t1_pmf_2d_final.sum()
                s2 = t2_pmf_2d_final.sum()
                if s1 > 0:
                    t1_pmf_2d_final /= s1
                if s2 > 0:
                    t2_pmf_2d_final /= s2

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Calculating win probability...")
                p_team1 = calculate_percentage_win_probability(t1_pmf_2d_final, t2_pmf_2d_final)
                p_team1_pct = p_team1 * 100.0
                p_team2_pct = 100.0 - p_team1_pct

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] ✓ {cat}: Team1 {p_team1_pct:.1f}% vs Team2 {p_team2_pct:.1f}%")

                t1_expected_ratio = expected_ratio_from_2d_pmf(t1_pmf_2d_final)
                t2_expected_ratio = expected_ratio_from_2d_pmf(t2_pmf_2d_final)

                mid_t1 = f"{t1_expected_ratio * 100:.1f}%"
                mid_t2 = f"{t2_expected_ratio * 100:.1f}%"

                if abs(p_team1_pct - p_team2_pct) < 0.5:
                    class_name = "is-tie"
                elif p_team1_pct > p_team2_pct:
                    class_name = "winner-left"
                else:
                    class_name = "winner-right"

                pmf1_data = compress_ratio_pmf_from_2d(t1_pmf_2d_final)
                pmf2_data = compress_ratio_pmf_from_2d(t2_pmf_2d_final)

                rows.append({
                    "cat": cat,
                    "p1": round(p_team1_pct, 1),
                    "p2": round(p_team2_pct, 1),
                    "class_name": class_name,
                    "mid_t1": mid_t1,
                    "mid_t2": mid_t2,
                    "pmf1": pmf1_data,
                    "pmf2": pmf2_data,
                })

            except Exception as e:
                if DEBUG_COMPARE_PRESENTER:
                    print(f"[ERROR] Failed to calculate {cat}: {e}")
                import traceback
                traceback.print_exc()
                rows.append({
                    "cat": cat,
                    "p1": 50.0,
                    "p2": 50.0,
                    "class_name": "is-tie",
                    "mid_t1": "-",
                    "mid_t2": "-",
                    "pmf1": {'min': 0, 'probs': []},
                    "pmf2": {'min': 0, 'probs': []},
                })
            continue

        # ----- Counting categories (1D PMFs) -----
        t1_stat = team1_current_stats.get(cat, {})
        t2_stat = team2_current_stats.get(cat, {})

        t1_current = t1_stat.get('value', 0.0) if isinstance(t1_stat, dict) else 0.0
        t2_current = t2_stat.get('value', 0.0) if isinstance(t2_stat, dict) else 0.0

        try:
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Calculating Team 1 PMF for {cat}...")
            t1_projected_pmf = calculate_team_pmf(data_team_players_1, cat)

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Calculating Team 2 PMF for {cat}...")
            t2_projected_pmf = calculate_team_pmf(data_team_players_2, cat)

            t1_current_int = int(round(t1_current))
            t2_current_int = int(round(t2_current))

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Adding current totals (T1: {t1_current_int}, T2: {t2_current_int})")

            t1_final_pmf = np.pad(t1_projected_pmf, (t1_current_int, 0))
            t2_final_pmf = np.pad(t2_projected_pmf, (t2_current_int, 0))

            # Trim tails before computing win probs & sending to frontend
            t1_final_pmf = trim_1d_pmf(t1_final_pmf)
            t2_final_pmf = trim_1d_pmf(t2_final_pmf)

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Calculating win probability...")

            if cat == 'TO':
                p_team1 = calculate_win_probability(t2_final_pmf, t1_final_pmf)
                p_team1 = 1.0 - p_team1
            else:
                p_team1 = calculate_win_probability(t1_final_pmf, t2_final_pmf)

            p_team1_pct = p_team1 * 100.0
            p_team2_pct = 100.0 - p_team1_pct

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] ✓ {cat}: Team1 {p_team1_pct:.1f}% vs Team2 {p_team2_pct:.1f}%")

            t1_indices = np.arange(len(t1_final_pmf))
            t2_indices = np.arange(len(t2_final_pmf))
            t1_expected = float(np.sum(t1_indices * t1_final_pmf))
            t2_expected = float(np.sum(t2_indices * t2_final_pmf))

            mid_t1 = str(round(t1_expected))
            mid_t2 = str(round(t2_expected))

            if abs(p_team1_pct - p_team2_pct) < 0.5:
                class_name = "is-tie"
            elif p_team1_pct > p_team2_pct:
                class_name = "winner-left"
            else:
                class_name = "winner-right"

            pmf1_data = compress_pmf(t1_final_pmf)
            pmf2_data = compress_pmf(t2_final_pmf)

            rows.append({
                "cat": cat,
                "p1": round(p_team1_pct, 1),
                "p2": round(p_team2_pct, 1),
                "class_name": class_name,
                "mid_t1": mid_t1,
                "mid_t2": mid_t2,
                "pmf1": pmf1_data,
                "pmf2": pmf2_data,
            })

        except Exception as e:
            if DEBUG_COMPARE_PRESENTER:
                print(f"[ERROR] Failed to calculate {cat}: {e}")
            import traceback
            traceback.print_exc()
            rows.append({
                "cat": cat,
                "p1": 50.0,
                "p2": 50.0,
                "class_name": "is-tie",
                "mid_t1": "-",
                "mid_t2": "-",
                "pmf1": {'min': 0, 'probs': []},
                "pmf2": {'min': 0, 'probs': []},
            })

    return rows
