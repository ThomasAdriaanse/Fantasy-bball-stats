"""
Compare presenter for fantasy basketball matchups.
Uses histogram-based convolution for exact win probability calculations.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np
import time

from .s3_service import load_player_dataset
from .classes.pmf1d import PMF1D
from .classes.pmf2d import PMF2D

from .PMF_utils import (
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

    # Always print at least once so we know this function is being hit
    print(f"[build_odds_rows] called (debug={DEBUG_COMPARE_PRESENTER})")

    t_start = time.perf_counter() if DEBUG_COMPARE_PRESENTER else None

    if DEBUG_COMPARE_PRESENTER:
        print("========== build_odds_rows ==========")
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
        rows = [
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
        if DEBUG_COMPARE_PRESENTER and t_start is not None:
            elapsed_total = time.perf_counter() - t_start
            print(f"[TIMING] build_odds_rows total={elapsed_total:.3f}s (dummy rows)")
        return rows

    rows: List[Dict[str, Any]] = []

    for cat in categories:
        cat_start = time.perf_counter() if DEBUG_COMPARE_PRESENTER else None

        if DEBUG_COMPARE_PRESENTER:
            print("\n----------------------------------------")
            print(f"[INFO] Calculating odds for: {cat}")

        # --------------------------------------------------
        # 1) Percentage categories (FG%, FT%) using PMF2D
        # --------------------------------------------------
        if cat in ('FG%', 'FT%'):
            try:
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

                makes_col, attempts_col = PERCENTAGE_CATEGORIES[cat]

                t1_pmf_2d_proj: PMF2D = build_team_pmf_2d(
                    data_team_players_1,
                    makes_col=makes_col,
                    attempts_col=attempts_col,
                    season=CURRENT_SEASON,
                    load_player_pmfs=load_player_pmfs,
                    debug=DEBUG_COMPARE_PRESENTER,
                )

                t2_pmf_2d_proj: PMF2D = build_team_pmf_2d(
                    data_team_players_2,
                    makes_col=makes_col,
                    attempts_col=attempts_col,
                    season=CURRENT_SEASON,
                    load_player_pmfs=load_player_pmfs,
                    debug=DEBUG_COMPARE_PRESENTER,
                )

                t1_made_i = int(round(t1_made))
                t1_att_i  = int(round(t1_att))
                t2_made_i = int(round(t2_made))
                t2_att_i  = int(round(t2_att))

                t1_pmf_2d_final = t1_pmf_2d_proj.shifted(t1_made_i, t1_att_i)
                t2_pmf_2d_final = t2_pmf_2d_proj.shifted(t2_made_i, t2_att_i)

                t1_pmf_2d_final.p[t1_pmf_2d_final.p < 1e-4] = 0
                t2_pmf_2d_final.p[t2_pmf_2d_final.p < 1e-4] = 0

                t1_pmf_2d_final.normalize()
                t2_pmf_2d_final.normalize()

                p_team1 = calculate_percentage_win_probability(
                    t1_pmf_2d_final,
                    t2_pmf_2d_final,
                )
                p_team1_pct = p_team1 * 100.0
                p_team2_pct = 100.0 - p_team1_pct

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

                if DEBUG_COMPARE_PRESENTER and cat_start is not None:
                    elapsed_cat = time.perf_counter() - cat_start
                    print(f"[TIMING] {cat} (% cat) took {elapsed_cat:.3f}s "
                          f"(T1 {p_team1_pct:.1f}% vs T2 {p_team2_pct:.1f}%)")

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

                if DEBUG_COMPARE_PRESENTER and cat_start is not None:
                    elapsed_cat = time.perf_counter() - cat_start
                    print(f"[TIMING] {cat} (% cat) failed after {elapsed_cat:.3f}s")

            continue

        # --------------------------------------------------
        # 2) Counting categories (1D PMFs via PMF1D)
        # --------------------------------------------------
        t1_stat = team1_current_stats.get(cat, {})
        t2_stat = team2_current_stats.get(cat, {})

        t1_current = t1_stat.get('value', 0.0) if isinstance(t1_stat, dict) else 0.0
        t2_current = t2_stat.get('value', 0.0) if isinstance(t2_stat, dict) else 0.0

        try:
            stat_col = CATEGORY_COLUMN_MAP.get(cat)
            t1_projected_pmf: PMF1D = build_team_pmf_counting(
                data_team_players_1,
                stat_col=stat_col,
                season=CURRENT_SEASON,
                load_player_pmfs=load_player_pmfs,
                debug=DEBUG_COMPARE_PRESENTER,
            )

            t2_projected_pmf: PMF1D = build_team_pmf_counting(
                data_team_players_2,
                stat_col=stat_col,
                season=CURRENT_SEASON,
                load_player_pmfs=load_player_pmfs,
                debug=DEBUG_COMPARE_PRESENTER,
            )

            t1_current_int = int(round(t1_current))
            t2_current_int = int(round(t2_current))

            t1_final_pmf = t1_projected_pmf.shifted(t1_current_int)
            t2_final_pmf = t2_projected_pmf.shifted(t2_current_int)

            # Trim the PMFs
            t1_final_pmf.p[t1_final_pmf.p < 1e-3] = 0
            t2_final_pmf.p[t2_final_pmf.p < 1e-3] = 0

            if cat == 'TO':
                p_t2_beats_t1 = t2_final_pmf.prob_beats(t1_final_pmf)
                p_team1 = 1.0 - p_t2_beats_t1
            else:
                p_team1 = t1_final_pmf.prob_beats(t2_final_pmf)

            p_team1_pct = p_team1 * 100.0
            p_team2_pct = 100.0 - p_team1_pct

            t1_expected = t1_final_pmf.mean()
            t2_expected = t2_final_pmf.mean()

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

            if DEBUG_COMPARE_PRESENTER and cat_start is not None:
                elapsed_cat = time.perf_counter() - cat_start
                print(f"[TIMING] {cat} (counting cat) took {elapsed_cat:.3f}s "
                      f"(T1 {p_team1_pct:.1f}% vs T2 {p_team2_pct:.1f}%, "
                      f"mid_t1={mid_t1}, mid_t2={mid_t2})")

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

            if DEBUG_COMPARE_PRESENTER and cat_start is not None:
                elapsed_cat = time.perf_counter() - cat_start
                print(f"[TIMING] {cat} (counting cat) failed after {elapsed_cat:.3f}s")

    if DEBUG_COMPARE_PRESENTER and t_start is not None:
        elapsed_total = time.perf_counter() - t_start
        print("\n========== build_odds_rows DONE ==========")
        print(f"[TIMING] build_odds_rows total={elapsed_total:.3f}s")

    return rows
