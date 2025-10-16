from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd

import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd  # kept for compatibility, not directly used here


def _team_category_averages(league, year: int, stat_window: str = 'projected') -> Dict[str, Any]:
    """
    Compute league-wide per-team category averages for a given ESPN stats window.

    Returns:
      {
        'categories': ['FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS'],
        'teams': [
          {
            'team_name': '…',
            'abbrev': '…',
            'cats': {'FG%': 0.48, 'FT%': 0.79, '3PM': 10.2, 'REB': 48.7, 'AST': 23.4, 'STL': 7.9, 'BLK': 4.2, 'TO': 13.7, 'PTS': 108.3},
            'z':     {'FG%': 0.50, 'FT%': -0.12, '3PM': 0.8, ...},   # z > 0 = better (including inverted for TO)
            'punts': ['FT%', 'AST']
          },
          ...
        ]
      }
    Notes:
      • Percentages are returned as proportions (e.g., 0.476), not 47.6.
      • We derive FG%/FT% from summed makes/attempts (more stable than mean of player %).
      • Z-scores invert TO so that lower TO => higher (better) z.
    """

    # Columns that cpd.get_team_player_data expects/returns
    cols: List[str] = [
        'player_name','min','fgm','fga','fg%','ftm','fta','ft%','threeptm',
        'reb','ast','stl','blk','turno','pts','inj','fpts','games'
    ]

    # Build records per team
    records: List[Dict[str, Any]] = []
    for ti, team in enumerate(league.teams):
        # Get strict-window player rows (no fallbacks)
        df = cpd.get_team_player_data(
            league=league,
            team_num=ti,
            columns=cols,
            year=year,
            league_scoring_rules={
                'fgm':0,'fga':0,'ftm':0,'fta':0,'threeptm':0,
                'reb':0,'ast':0,'stl':0,'blk':0,'turno':0,'pts':0
            },
            week_data=None,
            stat_window=stat_window
        )

        # Coerce numeric columns, ignore 'N/A'
        num_cols = ['fgm','fga','ftm','fta','threeptm','reb','ast','stl','blk','turno','pts']
        for c in num_cols + ['fg%','ft%']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Sum player averages to get a team per-game estimate
        s = df[num_cols].sum(numeric_only=True)

        # Robust percentage derivation from team makes/attempts
        fgm, fga, ftm, fta = float(s.get('fgm', 0.0)), float(s.get('fga', 0.0)), float(s.get('ftm', 0.0)), float(s.get('fta', 0.0))
        fg_pct = (fgm / fga) if fga else 0.0
        ft_pct = (ftm / fta) if fta else 0.0

        cats = {
            'FG%': round(fg_pct, 4),
            'FT%': round(ft_pct, 4),
            '3PM': round(float(s.get('threeptm', 0.0)), 2),
            'REB': round(float(s.get('reb', 0.0)), 2),
            'AST': round(float(s.get('ast', 0.0)), 2),
            'STL': round(float(s.get('stl', 0.0)), 2),
            'BLK': round(float(s.get('blk', 0.0)), 2),
            'TO' : round(float(s.get('turno', 0.0)), 2),
            'PTS': round(float(s.get('pts', 0.0)), 2),
        }

        records.append({
            'team_name': team.team_name,
            'abbrev': getattr(team, 'team_abbrev', None) or getattr(team, 'teamAbbrev', ''),
            'cats': cats
        })

    # Compute z-scores across teams per category
    categories = ['FG%','FT%','3PM','REB','AST','STL','BLK','TO','PTS']
    mat = {c: np.array([r['cats'][c] for r in records], dtype=float) for c in categories}

    # For TO, lower is better — invert for z so that low TO -> positive z
    for c in categories:
        arr = mat[c]
        arr_for_z = -arr if c == 'TO' else arr
        mu = float(np.mean(arr_for_z))
        sd = float(np.std(arr_for_z, ddof=0)) or 1.0
        zs = (arr_for_z - mu) / sd
        for i, r in enumerate(records):
            r.setdefault('z', {})[c] = round(float(zs[i]), 3)

    # Punt heuristic:
    #   - z <= -0.6  OR  team ranks in bottom 3 for that category
    for c in categories:
        vals = mat[c]
        order = np.argsort(vals)  # ascending by raw value
        if c != 'TO':
            bottom3_idx = set(order[:3])     # low values are bad
        else:
            bottom3_idx = set(order[-3:])    # high TO is bad

        for i, r in enumerate(records):
            z = r['z'][c]
            is_bottom = i in bottom3_idx
            if z <= -0.6 or is_bottom:
                r.setdefault('punts', []).append(c)

    # Normalize output (stable sort of punts)
    for r in records:
        r['punts'] = sorted(r.get('punts', []))

    return {'categories': categories, 'teams': records}
