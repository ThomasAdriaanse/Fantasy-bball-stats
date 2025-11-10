from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd  # compatibility


def _team_category_averages(league, year: int, stat_window: str = "projected") -> Dict[str, Any]:
    """
    Build per-team fantasy basketball category summaries and league-relative strength comparisons.

    Returns:
      {
        'categories': [...],
        'teams': [
          {
            'team_name': '…',
            'abbrev': '…',
            'cats': {...},                  # raw totals (FG%, FT%, etc.)
            'diff_from_avg': {...},         # raw difference from league mean (TO inverted)
            'rank': {...},                  # 1 = best, N = worst (TO ranked inversely)
            'percentile': {...},            # 0..1 relative performance
            'strength_scaled': {...},       # min-max scaled 0..1 strength
            'punts': ['FG%', 'AST', ...],   # punted cats (very close to worst)
          },
          ...
        ]
      }
    """

    categories = ['FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS']
    cols = [
        'player_name', 'min', 'fgm', 'fga', 'fg%', 'ftm', 'fta', 'ft%', 'threeptm',
        'reb', 'ast', 'stl', 'blk', 'turno', 'pts', 'inj', 'fpts', 'games'
    ]

    team_records: List[Dict[str, Any]] = []
    volume_meta: List[Dict[str, float]] = []  # tracks each team’s FGA/FTA for volume-based logic

    # === Collect raw category totals per team ===
    for team_index, team in enumerate(league.teams):
        df = cpd.get_team_player_data(
            league=league,
            team_num=team_index,
            columns=cols,
            year=year,
            league_scoring_rules={
                'fgm': 0, 'fga': 0, 'ftm': 0, 'fta': 0,
                'threeptm': 0, 'reb': 0, 'ast': 0,
                'stl': 0, 'blk': 0, 'turno': 0, 'pts': 0
            },
            week_data=None,
            stat_window=stat_window,
        )

        # Convert numeric columns
        for c in ['fgm', 'fga', 'ftm', 'fta', 'threeptm', 'reb', 'ast',
                  'stl', 'blk', 'turno', 'pts', 'fg%', 'ft%']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        totals = df[['fgm', 'fga', 'ftm', 'fta', 'threeptm', 'reb', 'ast',
                     'stl', 'blk', 'turno', 'pts']].sum(numeric_only=True)

        # Compute team FG% and FT%
        fgm, fga = float(totals.get('fgm', 0)), float(totals.get('fga', 0))
        ftm, fta = float(totals.get('ftm', 0)), float(totals.get('fta', 0))
        fg_pct = (fgm / fga) if fga else 0.0
        ft_pct = (ftm / fta) if fta else 0.0

        # Record team totals
        cat_values = {
            'FG%': round(fg_pct, 4),
            'FT%': round(ft_pct, 4),
            '3PM': round(float(totals.get('threeptm', 0)), 2),
            'REB': round(float(totals.get('reb', 0)), 2),
            'AST': round(float(totals.get('ast', 0)), 2),
            'STL': round(float(totals.get('stl', 0)), 2),
            'BLK': round(float(totals.get('blk', 0)), 2),
            'TO':  round(float(totals.get('turno', 0)), 2),
            'PTS': round(float(totals.get('pts', 0)), 2),
        }

        team_records.append({
            'team_name': team.team_name,
            'abbrev': getattr(team, 'team_abbrev', None) or getattr(team, 'teamAbbrev', ''),
            'cats': cat_values,
        })
        volume_meta.append({'FGA': fga, 'FTA': fta})

    # === Build arrays per category for league-wide comparison ===
    league_vals = {c: np.array([t['cats'][c] for t in team_records], dtype=float)
                   for c in categories}

    # Adjust TO so “higher = better” for comparison
    adjusted_vals = {
        c: (-league_vals[c] if c == 'TO' else league_vals[c])
        for c in categories
    }

    # Store per-category min/max so we can reuse for punts
    cat_min: Dict[str, float] = {}
    cat_max: Dict[str, float] = {}

    # === Compute league-relative metrics per category ===
    for cat in categories:
        arr = adjusted_vals[cat]

        # Simple difference from league mean (raw performance difference)
        mean_val = float(arr.mean())
        diff_from_avg = arr - mean_val

        # Rank (1 = best), percentile (0..1), and min-max scaling (0..1)
        order = np.argsort(arr)  # ascending
        rank_tmp = np.empty_like(order)
        rank_tmp[order] = np.arange(1, len(arr) + 1)
        rank_visible = len(arr) - rank_tmp + 1  # flip so 1 = best
        percentile = rank_tmp / float(len(arr))

        min_val, max_val = float(arr.min()), float(arr.max())
        cat_min[cat] = min_val
        cat_max[cat] = max_val

        if max_val > min_val:
            strength_scaled = (arr - min_val) / (max_val - min_val)
        else:
            strength_scaled = np.full_like(arr, 0.5, dtype=float)

        # Store metrics for each team
        for i, t in enumerate(team_records):
            t.setdefault('diff_from_avg', {})[cat] = round(float(diff_from_avg[i]), 3)
            t.setdefault('rank', {})[cat] = int(rank_visible[i])
            t.setdefault('percentile', {})[cat] = round(float(percentile[i]), 3)
            t.setdefault('strength_scaled', {})[cat] = round(float(strength_scaled[i]), 3)

    # === Identify punt categories: only worst or within 5% of worst (range-based) ===
    fga_all = np.array([v['FGA'] for v in volume_meta], dtype=float)
    fta_all = np.array([v['FTA'] for v in volume_meta], dtype=float)
    fga_med = float(np.median(fga_all) or 0.0)
    fta_med = float(np.median(fta_all) or 0.0)

    for i, t in enumerate(team_records):
        punts: List[str] = []

        for cat in categories:
            arr = adjusted_vals[cat]
            min_val = cat_min[cat]
            max_val = cat_max[cat]
            span = max_val - min_val

            # if no spread, nobody is clearly punting this category
            if span <= 0:
                continue

            threshold = min_val + 0.05 * span  # 5% of worst→best range above worst
            this_val = float(arr[i])

            # volume gating for FG% / FT% so tiny-volume teams don't auto-punt
            vol_ok = True
            if cat == 'FG%':
                vol_ok = (volume_meta[i]['FGA'] >= 0.6 * fga_med)
            elif cat == 'FT%':
                vol_ok = (volume_meta[i]['FTA'] >= 0.6 * fta_med)

            if not vol_ok:
                continue

            # punt if at or very close to worst (within 5% of span from worst)
            if this_val <= threshold:
                punts.append(cat)

        t['punts'] = punts

    return {
        'categories': categories,
        'teams': team_records,
    }
