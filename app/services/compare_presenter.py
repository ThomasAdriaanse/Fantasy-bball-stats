"""
Compare presenter for fantasy basketball matchups.
Uses histogram-based convolution for exact win probability calculations.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np

from .s3_service import load_player_dataset
from .classes.pmf1d import PMF1D
from .classes.pmf2d import PMF2D

from .PMF_utils import (
    trim_1d_pmf,
    compress_pmf,
    build_team_pmf_counting,
    build_team_pmf_2d,
    compress_ratio_pmf_from_2d,
    expected_ratio_from_2d_pmf,
    calculate_percentage_win_probability,
    load_player_pmfs,
)



DEBUG_COMPARE_PRESENTER = True

CATEGORY_COLUMN_MAP = {
    'PTS': 'PTS',
    'REB': 'REB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    '3PM': 'FG3M',
    'TO':  'TOV',
}

PERCENTAGE_CATEGORIES = {
    'FG%': ('FGM', 'FGA'),
    'FT%': ('FTM', 'FTA'),
}

CURRENT_SEASON = "2025-26"


# ========= SNAPSHOT ROWS =========

def build_snapshot_rows(
    team1_current_stats: Optional[Dict[str, Any]],
    team2_current_stats: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
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
    Produces one row per category with:
      - win odds for each team
      - an expected mid value
      - compressed PMF data for the frontend graphs
    """
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%', 'TO']

    if DEBUG_COMPARE_PRESENTER:
        print("========== build_odds_rows DEBUG ==========")
        print("  team1_current_stats keys:", list((team1_current_stats or {}).keys()))
        print("  team2_current_stats keys:", list((team2_current_stats or {}).keys()))
        print("  data_team_players_1 len:",
              len(data_team_players_1) if data_team_players_1 else 0)
        print("  data_team_players_2 len:",
              len(data_team_players_2) if data_team_players_2 else 0)

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

    rows: List[Dict[str, Any]] = []

    for cat in categories:
        if DEBUG_COMPARE_PRESENTER:
            print("\n----------------------------------------")
            print(f"[INFO] Calculating odds for: {cat}")

        # --------------------------------------------------
        # 1) Percentage categories (FG%, FT%) using PMF2D
        # --------------------------------------------------
        if cat in ('FG%', 'FT%'):
            try:
                # 1.1) Extract current totals (made / attempted) for both teams
                if cat == 'FG%':
                    t1_made = float(team1_current_stats.get('FGM', {}).get('value', 0.0))
                    t1_att  = float(team1_current_stats.get('FGA', {}).get('value', 0.0))
                    t2_made = float(team2_current_stats.get('FGM', {}).get('value', 0.0))
                    t2_att  = float(team2_current_stats.get('FGA', {}).get('value', 0.0))
                else:  # FT%
                    t1_made = float(team1_current_stats.get('FTM', {}).get('value', 0.0))
                    t1_att  = float(team1_current_stats.get('FTA', {}).get('value', 0.0))
                    t2_made = float(team2_current_stats.get('FTM', {}).get('value', 0.0))
                    t2_att  = float(team2_current_stats.get('FTA', {}).get('value', 0.0))

                if DEBUG_COMPARE_PRESENTER:
                    print(f"[FT%/FG% DEBUG] current totals:")
                    print(f"  T1: made={t1_made}, att={t1_att}")
                    print(f"  T2: made={t2_made}, att={t2_att}")

                makes_col, attempts_col = PERCENTAGE_CATEGORIES[cat]

                # 1.2) Build projected 2D PMFs (makes/attempts) for each team
                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Building Team 1 projected PMF2D for {cat} "
                          f"using load_player_pmfs...")
                t1_pmf_2d_proj: PMF2D = build_team_pmf_2d(
                    data_team_players_1,
                    makes_col=makes_col,
                    attempts_col=attempts_col,
                    season=CURRENT_SEASON,
                    load_player_pmfs=load_player_pmfs,
                    debug=DEBUG_COMPARE_PRESENTER,
                )

                if DEBUG_COMPARE_PRESENTER:
                    proj_arr_t1 = t1_pmf_2d_proj.p
                    print(f"    T1 projected PMF2D shape={proj_arr_t1.shape}, "
                          f"sum={proj_arr_t1.sum():.6f}")

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Building Team 2 projected PMF2D for {cat} "
                          f"using load_player_pmfs...")
                t2_pmf_2d_proj: PMF2D = build_team_pmf_2d(
                    data_team_players_2,
                    makes_col=makes_col,
                    attempts_col=attempts_col,
                    season=CURRENT_SEASON,
                    load_player_pmfs=load_player_pmfs,
                    debug=DEBUG_COMPARE_PRESENTER,
                )

                if DEBUG_COMPARE_PRESENTER:
                    proj_arr_t2 = t2_pmf_2d_proj.p
                    print(f"    T2 projected PMF2D shape={proj_arr_t2.shape}, "
                          f"sum={proj_arr_t2.sum():.6f}")

                # Some sanity: expected ratios BEFORE adding current totals
                if DEBUG_COMPARE_PRESENTER:
                    try:
                        t1_proj_ratio = expected_ratio_from_2d_pmf(t1_pmf_2d_proj)
                        t2_proj_ratio = expected_ratio_from_2d_pmf(t2_pmf_2d_proj)
                        print(f"    T1 projected ratio (no current) ~ {t1_proj_ratio*100:.2f}%")
                        print(f"    T2 projected ratio (no current) ~ {t2_proj_ratio*100:.2f}%")
                    except Exception as e:
                        print(f"    [DEBUG] Expected ratio on projected PMFs failed: {e}")

                # 1.3) Add current totals into the projected distributions via shift
                t1_made_i = int(round(t1_made))
                t1_att_i  = int(round(t1_att))
                t2_made_i = int(round(t2_made))
                t2_att_i  = int(round(t2_att))

                if DEBUG_COMPARE_PRESENTER:
                    print("  [ODDS-2D] Shifting by current totals:")
                    print(f"    T1 shift: ({t1_made_i}, {t1_att_i})")
                    print(f"    T2 shift: ({t2_made_i}, {t2_att_i})")

                t1_pmf_2d_final = t1_pmf_2d_proj.shifted(t1_made_i, t1_att_i)
                t2_pmf_2d_final = t2_pmf_2d_proj.shifted(t2_made_i, t2_att_i)

                # extra safety normalize
                t1_pmf_2d_final.normalize()
                t2_pmf_2d_final.normalize()

                if DEBUG_COMPARE_PRESENTER:
                    final_arr_t1 = t1_pmf_2d_final.p
                    final_arr_t2 = t2_pmf_2d_final.p
                    print(f"    T1 final PMF2D shape={final_arr_t1.shape}, "
                          f"sum={final_arr_t1.sum():.6f}")
                    print(f"    T2 final PMF2D shape={final_arr_t2.shape}, "
                          f"sum={final_arr_t2.sum():.6f}")

                # 1.4) Compute win probability in % space
                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Calculating win probability in % space...")
                p_team1 = calculate_percentage_win_probability(
                    t1_pmf_2d_final,
                    t2_pmf_2d_final,
                )
                p_team1_pct = p_team1 * 100.0
                p_team2_pct = 100.0 - p_team1_pct

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Result {cat}: Team1 {p_team1_pct:.1f}% "
                          f"vs Team2 {p_team2_pct:.1f}%")

                # 1.5) Expected ratios for mid markers
                t1_expected_ratio = expected_ratio_from_2d_pmf(t1_pmf_2d_final)
                t2_expected_ratio = expected_ratio_from_2d_pmf(t2_pmf_2d_final)

                mid_t1 = f"{t1_expected_ratio * 100:.1f}%"
                mid_t2 = f"{t2_expected_ratio * 100:.1f}%"

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] mid_t1={mid_t1}, mid_t2={mid_t2}")

                # 1.6) Determine CSS class
                if abs(p_team1_pct - p_team2_pct) < 0.5:
                    class_name = "is-tie"
                elif p_team1_pct > p_team2_pct:
                    class_name = "winner-left"
                else:
                    class_name = "winner-right"

                # 1.7) Compress PMFs for frontend charts
                pmf1_data = compress_ratio_pmf_from_2d(t1_pmf_2d_final)
                pmf2_data = compress_ratio_pmf_from_2d(t2_pmf_2d_final)

                if DEBUG_COMPARE_PRESENTER:
                    print(f"  [ODDS-2D] Compressed PMF1: min={pmf1_data.get('min')}, "
                          f"len={len(pmf1_data.get('probs', []))}")
                    print(f"  [ODDS-2D] Compressed PMF2: min={pmf2_data.get('min')}, "
                          f"len={len(pmf2_data.get('probs', []))}")

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
                    print(f"[ERROR] Failed to calculate {cat} (percentage cat): {e}")
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

        # --------------------------------------------------
        # 2) Counting categories (1D PMFs via PMF1D)
        # --------------------------------------------------
        t1_stat = team1_current_stats.get(cat, {})
        t2_stat = team2_current_stats.get(cat, {})

        t1_current = t1_stat.get('value', 0.0) if isinstance(t1_stat, dict) else 0.0
        t2_current = t2_stat.get('value', 0.0) if isinstance(t2_stat, dict) else 0.0

        if DEBUG_COMPARE_PRESENTER:
            print(f"  [ODDS] Current totals {cat}: "
                  f"T1={t1_current} T2={t2_current}")

        try:
            # 2.1) Build projected team PMFs (future games only)
            stat_col = CATEGORY_COLUMN_MAP.get(cat)
            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Building Team 1 projected PMF1D for {cat} "
                      f"using load_player_pmfs (stat_col={stat_col})...")
            t1_projected_pmf: PMF1D = build_team_pmf_counting(
                data_team_players_1,
                stat_col=stat_col,
                season=CURRENT_SEASON,
                load_player_pmfs=load_player_pmfs,
                debug=DEBUG_COMPARE_PRESENTER,
            )

            if DEBUG_COMPARE_PRESENTER:
                arr_t1_proj = t1_projected_pmf.p
                print(f"    T1 projected PMF size={arr_t1_proj.size}, sum={arr_t1_proj.sum():.6f}")
                print(f"    T1 projected mean (no current)={t1_projected_pmf.mean():.3f}")

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Building Team 2 projected PMF1D for {cat} "
                      f"using load_player_pmfs (stat_col={stat_col})...")
            t2_projected_pmf: PMF1D = build_team_pmf_counting(
                data_team_players_2,
                stat_col=stat_col,
                season=CURRENT_SEASON,
                load_player_pmfs=load_player_pmfs,
                debug=DEBUG_COMPARE_PRESENTER,
            )

            if DEBUG_COMPARE_PRESENTER:
                arr_t2_proj = t2_projected_pmf.p
                print(f"    T2 projected PMF size={arr_t2_proj.size}, sum={arr_t2_proj.sum():.6f}")
                print(f"    T2 projected mean (no current)={t2_projected_pmf.mean():.3f}")

            # 2.2) Shift by current totals to get full-week totals
            t1_current_int = int(round(t1_current))
            t2_current_int = int(round(t2_current))

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Shifting PMFs by current totals for {cat}: "
                      f"T1 shift={t1_current_int}, T2 shift={t2_current_int}")

            t1_final_pmf = t1_projected_pmf.shifted(t1_current_int)
            t2_final_pmf = t2_projected_pmf.shifted(t2_current_int)

            if DEBUG_COMPARE_PRESENTER:
                print(f"    T1 mean after shift={t1_final_pmf.mean():.3f}")
                print(f"    T2 mean after shift={t2_final_pmf.mean():.3f}")

            # 2.3) Trim tails before computing win probs / sending to frontend
            t1_final_pmf = trim_1d_pmf(t1_final_pmf)
            t2_final_pmf = trim_1d_pmf(t2_final_pmf)

            if DEBUG_COMPARE_PRESENTER:
                arr_t1_final = t1_final_pmf.p
                arr_t2_final = t2_final_pmf.p
                nz1 = np.where(arr_t1_final > 0)[0]
                nz2 = np.where(arr_t2_final > 0)[0]
                print(f"    T1 final support: "
                      f"[{nz1[0]}..{nz1[-1]}], size={arr_t1_final.size}, sum={arr_t1_final.sum():.6f}")
                print(f"    T2 final support: "
                      f"[{nz2[0]}..{nz2[-1]}], size={arr_t2_final.size}, sum={arr_t2_final.sum():.6f}")

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Calculating win probability for {cat}...")

            # 2.4) Compute win probability
            if cat == 'TO':
                # lower is better for turnovers
                p_t2_beats_t1 = t2_final_pmf.prob_beats(t1_final_pmf)
                p_team1 = 1.0 - p_t2_beats_t1
            else:
                p_team1 = t1_final_pmf.prob_beats(t2_final_pmf)

            p_team1_pct = p_team1 * 100.0
            p_team2_pct = 100.0 - p_team1_pct

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Result {cat}: Team1 {p_team1_pct:.1f}% "
                      f"vs Team2 {p_team2_pct:.1f}%")

            # 2.5) Compute expected value (mean) of each PMF for mid-marker
            t1_expected = t1_final_pmf.mean()
            t2_expected = t2_final_pmf.mean()

            mid_t1 = str(round(t1_expected))
            mid_t2 = str(round(t2_expected))

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] mid_t1={mid_t1}, mid_t2={mid_t2}")

            # 2.6) Determine which side wins visually
            if abs(p_team1_pct - p_team2_pct) < 0.5:
                class_name = "is-tie"
            elif p_team1_pct > p_team2_pct:
                class_name = "winner-left"
            else:
                class_name = "winner-right"

            # 2.7) Compress PMFs into {min, probs[]} for frontend charts
            pmf1_data = compress_pmf(t1_final_pmf)
            pmf2_data = compress_pmf(t2_final_pmf)

            if DEBUG_COMPARE_PRESENTER:
                print(f"  [ODDS] Compressed PMF1 for {cat}: "
                      f"min={pmf1_data.get('min')}, "
                      f"len={len(pmf1_data.get('probs', []))}")
                print(f"  [ODDS] Compressed PMF2 for {cat}: "
                      f"min={pmf2_data.get('min')}, "
                      f"len={len(pmf2_data.get('probs', []))}")

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
                print(f"[ERROR] Failed to calculate {cat} (counting cat): {e}")
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

    if DEBUG_COMPARE_PRESENTER:
        print("\n========== build_odds_rows DONE ==========")

    return rows
