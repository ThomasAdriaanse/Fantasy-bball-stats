"""
PMF utilities (1D and 2D)

- Build PMFs from game logs (1D).
- Compress / trim PMFs for UI.
- 2D PMFs for FG% / FT%.
- Team-level PMF builders using an injected loader.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, List, Sequence, Tuple, Optional
import time

import numpy as np
from scipy import signal

from .classes.pmf1d import PMF1D
from .classes.pmf2d import PMF2D

# Tail trimming params for 1D PMFs
TAIL_MASS_1D = 0.003
MIN_BINS_AFTER_TRIM = 10

# For ratio PMFs
TAIL_MASS_RATIO = 0.003
RATIO_BUCKETS = 1001  # 0..1000 -> 0.0..100.0%


# ========= 1D HELPERS (USING PMF1D) =========

def build_pmf_from_games(stat_values: np.ndarray) -> PMF1D:
    """
    Build a 1D PMF from a sequence of game stats (counts).

    Returns a PMF1D over non-negative integers.
    """
    if len(stat_values) == 0:
        return PMF1D(np.array([1.0]))  # delta at 0

    stat_values = np.round(stat_values).astype(int)
    stat_values = np.clip(stat_values, 0, None)

    max_val = int(stat_values.max())
    counts = np.bincount(stat_values, minlength=max_val + 1).astype(float)
    total = counts.sum()
    if total > 0:
        counts /= total

    return PMF1D(counts)


def convolve_pmf_n_times(pmf: PMF1D, n: int) -> PMF1D:
    """
    Convolve a PMF with itself n times (n i.i.d. games).
    """
    if n <= 0:
        return PMF1D(np.array([1.0]))  # delta at 0
    if n == 1:
        return pmf.copy()

    result = pmf.copy()
    for _ in range(n - 1):
        result = result.convolve(pmf)
    return result


def trim_1d_pmf(
    pmf: PMF1D,
    tail_mass: float = TAIL_MASS_1D,
    min_bins: int = MIN_BINS_AFTER_TRIM,
) -> PMF1D:
    """
    Soft-trim extreme tails from a 1D PMF by zeroing out very small
    mass at both ends, then renormalizing.

    - Keeps the array length the same (indices still mean the same totals).
    - Uses a quantile-based cutoff so trimming adapts to the shape.
    """
    arr = pmf.p.copy()
    if arr.size == 0:
        return pmf

    total = arr.sum()
    if total <= 0:
        return pmf

    norm = arr / total
    cdf = np.cumsum(norm)

    low_idx = int(np.searchsorted(cdf, tail_mass, side="left"))
    high_idx = int(np.searchsorted(cdf, 1.0 - tail_mass, side="right") - 1)

    if high_idx - low_idx + 1 < min_bins:
        return pmf

    trimmed = np.zeros_like(arr)
    trimmed[low_idx:high_idx + 1] = arr[low_idx:high_idx + 1]

    new_total = trimmed.sum()
    if new_total <= 0:
        return pmf

    trimmed /= new_total
    return PMF1D(trimmed)


def calculate_win_probability(team1_pmf: PMF1D, team2_pmf: PMF1D) -> float:
    """
    P(Team1 > Team2).
    """
    return team1_pmf.prob_beats(team2_pmf)


def compress_pmf(pmf: PMF1D) -> Dict[str, Any]:
    """
    Compress a 1D PMF into {'min': int, 'probs': [floats]} after trimming tails.
    """
    trimmed = trim_1d_pmf(pmf)
    arr = trimmed.p

    if arr.size == 0:
        return {"min": 0, "probs": []}

    indices = np.where(arr > 0)[0]
    if len(indices) == 0:
        return {"min": 0, "probs": []}

    start_idx = int(indices[0])
    end_idx = int(indices[-1])
    probs = [round(float(p), 6) for p in arr[start_idx:end_idx + 1]]

    return {"min": start_idx, "probs": probs}


# ========= 2D PMF HELPERS (USING PMF2D) =========

def build_2d_pmf_from_games(makes: np.ndarray, attempts: np.ndarray) -> PMF2D:
    """
    Build a single-game 2D PMF over (makes, attempts).

    pmf[m, a] = probability of 'makes = m' and 'attempts = a' in a game.
    """
    if len(makes) == 0 or len(attempts) == 0:
        base = np.zeros((1, 1), dtype=float)
        base[0, 0] = 1.0
        return PMF2D(base)

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

    if pmf.sum() == 0:
        pmf[0, 0] = 1.0

    return PMF2D(pmf)


def convolve_pmf_2d_n_times(pmf2d: PMF2D, n: int) -> PMF2D:
    """
    Convolve a 2D PMF with itself n times (e.g., n games of FG/FT attempts).
    """
    if n <= 0:
        base = np.zeros((1, 1), dtype=float)
        base[0, 0] = 1.0
        return PMF2D(base)
    if n == 1:
        return pmf2d.copy()

    result = pmf2d.copy()
    for _ in range(n - 1):
        result = result.convolve(pmf2d)
    return result


def _ratios_and_probs_from_2d_pmf(pmf2d: PMF2D) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten 2D pmf[m,a] into (ratios, probs), ratio = m/a (0 if a=0).
    """
    arr = pmf2d.p
    m_idx, a_idx = np.nonzero(arr)
    if len(m_idx) == 0:
        return np.array([0.0]), np.array([1.0])

    probs = arr[m_idx, a_idx]
    ratios = np.zeros_like(probs, dtype=float)
    mask = a_idx > 0
    ratios[mask] = m_idx[mask] / a_idx[mask]

    s = probs.sum()
    if s > 0:
        probs = probs / s
    return ratios, probs


def compress_ratio_pmf_from_2d(pmf2d: PMF2D) -> Dict[str, Any]:
    """
    Compress a 2D (makes, attempts) PMF into a 1D PMF over percentage
    points with finer resolution, formatted as {'min': int, 'probs': [floats]}.
    """
    ratios, probs = _ratios_and_probs_from_2d_pmf(pmf2d)  # ratios in [0,1]

    if len(ratios) == 0:
        return {'min': 0, 'probs': []}

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

    low_idx = int(np.searchsorted(cdf, TAIL_MASS_RATIO, side="left"))
    high_idx = int(np.searchsorted(cdf, 1.0 - TAIL_MASS_RATIO, side="right") - 1)

    if high_idx - low_idx + 1 < MIN_BINS_AFTER_TRIM:
        indices = np.where(buckets > 1e-4)[0]
        if len(indices) == 0:
            return {'min': 0, 'probs': []}
        start = int(indices[0])
        end = int(indices[-1])
    else:
        start = low_idx
        end = high_idx

    probs_list = [round(float(p), 4) for p in norm[start:end + 1]]
    return {'min': start, 'probs': probs_list}


def expected_ratio_from_2d_pmf(pmf2d: PMF2D) -> float:
    """
    E[ratio] from 2D PMF.

    Thin wrapper around PMF2D.expected_ratio() for call-site symmetry.
    """
    return pmf2d.expected_ratio()


def calculate_percentage_win_probability(
    team1_pmf2d: PMF2D,
    team2_pmf2d: PMF2D,
) -> float:
    """
    P(Team1% > Team2%) using ratio distributions from 2D PMFs.
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


# ========= TEAM-LEVEL BUILDERS (GENERIC, REUSABLE FOR PROJECT) =========

def build_team_pmf_counting(
    team_players: List[Dict[str, Any]],
    *,
    stat_col: Optional[str],
    season: str,
    load_player_df: Callable[[str], Any],
    games_field: str = "games",
    injury_field: str = "inj",
    skip_injury_status: Sequence[str] = ("OUT", "INJURY_RESERVE", "IR", "SUSPENDED"),
    debug: bool = False,
) -> PMF1D:
    """
    Build a team-level 1D PMF for a counting stat (PTS, REB, etc.)
    """
    if not stat_col:
        return PMF1D(np.array([1.0]))  # delta at 0

    team_pmf = PMF1D(np.array([1.0]))  # start as delta at 0
    start_time = time.time()
    player_count = 0

    if debug:
        print(f"  [PMF] Processing {len(team_players)} players for {stat_col}")

    for player in team_players:
        player_name = player.get("player_name")
        games_remaining = int(player.get(games_field, 0))

        if not player_name or games_remaining <= 0:
            continue

        injury_status = str(player.get(injury_field, "ACTIVE"))
        if injury_status in skip_injury_status:
            if debug:
                print(f"  [PMF] Skipping {player_name} (injured: {injury_status})")
            continue

        player_count += 1
        player_start = time.time()
        if debug:
            print(f"  [PMF] [{player_count}] {player_name} ({games_remaining} games)")

        load_start = time.time()
        df = load_player_df(player_name)
        load_time = time.time() - load_start

        if df is None or df.empty:
            if debug:
                print(f"  [PMF] ⚠️  No data for {player_name} ({load_time:.2f}s)")
            continue

        if "SEASON" in df.columns:
            current_year = int(season.split("-")[0])
            df = df[
                df["SEASON"].str.startswith(str(current_year))
                | df["SEASON"].str.startswith(str(current_year + 1))
            ]

        if df.empty or stat_col not in df.columns:
            if debug:
                print(f"  [PMF] ⚠️  No {stat_col} data for {player_name}")
            continue

        values = df[stat_col].dropna().values
        if len(values) == 0:
            if debug:
                print(f"  [PMF] ⚠️  No valid {stat_col} values for {player_name}")
            continue

        pmf_start = time.time()
        single_game_pmf = build_pmf_from_games(values)
        pmf_time = time.time() - pmf_start

        conv_start = time.time()
        total_pmf = convolve_pmf_n_times(single_game_pmf, games_remaining)
        conv_time = time.time() - conv_start

        combine_start = time.time()
        team_pmf = team_pmf.convolve(total_pmf)
        combine_time = time.time() - combine_start

        if debug:
            player_total = time.time() - player_start
            print(
                f"  [PMF]   ✓ {player_total:.2f}s "
                f"(load:{load_time:.2f}s pmf:{pmf_time:.3f}s conv:{conv_time:.3f}s "
                f"combine:{combine_time:.3f}s)"
            )

    if debug:
        total_time = time.time() - start_time
        print(
            f"  [PMF] ✓ Completed {stat_col} in {total_time:.2f}s "
            f"- Final team PMF support size: {team_pmf.size}"
        )
    return team_pmf


def build_team_pmf_2d(
    team_players: List[Dict[str, Any]],
    *,
    makes_col: str,
    attempts_col: str,
    season: str,
    load_player_df: Callable[[str], Any],
    games_field: str = "games",
    injury_field: str = "inj",
    skip_injury_status: Sequence[str] = ("OUT", "INJURY_RESERVE", "IR", "SUSPENDED"),
    debug: bool = False,
) -> PMF2D:
    """
    Build a team-level 2D PMF for (makes, attempts) stats (FG, FT).

    Returns:
        PMF2D where p[m, a] = P(total makes=m, attempts=a) for remaining games.
    """
    start_time = time.time()
    team_pmf = PMF2D(np.array([[1.0]]))  # delta at (0,0)
    player_count = 0

    if debug:
        print(f"  [PMF-2D] Processing {len(team_players)} players for {makes_col}/{attempts_col}")

    for player in team_players:
        player_name = player.get("player_name")
        games_remaining = int(player.get(games_field, 0))
        if not player_name or games_remaining <= 0:
            continue

        injury_status = str(player.get(injury_field, "ACTIVE"))
        if injury_status in skip_injury_status:
            if debug:
                print(f"  [PMF-2D] Skipping {player_name} (injured: {injury_status})")
            continue

        player_count += 1
        player_start = time.time()
        if debug:
            print(f"  [PMF-2D] [{player_count}] {player_name} ({games_remaining} games)")

        load_start = time.time()
        df = load_player_df(player_name)
        load_time = time.time() - load_start

        if df is None or df.empty:
            if debug:
                print(f"  [PMF-2D] ⚠️  No data for {player_name} ({load_time:.2f}s)")
            continue

        if "SEASON" in df.columns:
            current_year = int(season.split("-")[0])
            df = df[
                df["SEASON"].str.startswith(str(current_year))
                | df["SEASON"].str.startswith(str(current_year + 1))
            ]

        if df.empty or makes_col not in df.columns or attempts_col not in df.columns:
            if debug:
                print(f"  [PMF-2D] ⚠️  No {makes_col}/{attempts_col} data for {player_name}")
            continue

        makes = df[makes_col].dropna().values
        attempts = df[attempts_col].dropna().values
        if len(makes) == 0 or len(attempts) == 0:
            if debug:
                print(f"  [PMF-2D] ⚠️  No valid {makes_col}/{attempts_col} values")
            continue

        pmf_start = time.time()
        single_2d = build_2d_pmf_from_games(makes, attempts)
        pmf_time = time.time() - pmf_start

        conv_start = time.time()
        total_2d = convolve_pmf_2d_n_times(single_2d, games_remaining)
        conv_time = time.time() - conv_start

        combine_start = time.time()
        team_pmf = team_pmf.convolve(total_2d)
        combine_time = time.time() - combine_start

        if debug:
            player_total = time.time() - player_start
            print(
                f"  [PMF-2D]   ✓ {player_total:.2f}s "
                f"(load:{load_time:.2f}s pmf:{pmf_time:.3f}s "
                f"conv:{conv_time:.3f}s combine:{combine_time:.3f}s) "
                f"PMF shape: {team_pmf.shape}"
            )

    if debug:
        total_time = time.time() - start_time
        print(
            f"  [PMF-2D] ✓ Completed {makes_col}/{attempts_col} in {total_time:.2f}s "
            f"- Final PMF shape: {team_pmf.shape}"
        )
    return team_pmf
