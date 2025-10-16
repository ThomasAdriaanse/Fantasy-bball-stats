from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd  # compatibility


def _team_category_averages(league, year: int, stat_window: str = "projected") -> Dict[str, Any]:
    """
    Returns:
      {
        'categories': [...],
        'teams': [
          {
            'team_name': '…',
            'abbrev': '…',
            # Raw values (FG%/FT% in proportions)
            'cats': {'FG%': 0.48, 'FT%': 0.79, '3PM': ..., 'TO': ..., ...},

            # League-relative analytics per cat:
            'z': {'FG%': 0.41, 'TO': 0.55, ...},                # z>0 is better (TO inverted)
            'rank': {'FG%': 4, 'TO': 3, ...},                   # 1 = best (TO ranks use “low TO is best”)
            'pct': {'FG%': 0.72, 'TO': 0.81, ...},              # percentile 0..1 where higher = better (TO inverted)
            'strength': {'FG%': 0.76, 'TO': 0.84, ...},         # min-max 0..1 for bar length (higher = better)
            'punts': ['FG%', 'AST']                             # stricter heuristic below
          },
          ...
        ]
      }
    """
    categories = ['FG%','FT%','3PM','REB','AST','STL','BLK','TO','PTS']

    cols = [
        'player_name','min','fgm','fga','fg%','ftm','fta','ft%','threeptm',
        'reb','ast','stl','blk','turno','pts','inj','fpts','games'
    ]

    records: List[Dict[str, Any]] = []
    vol_meta: List[Dict[str, float]] = []  # for % volume (attempts)

    for ti, team in enumerate(league.teams):
        df = cpd.get_team_player_data(
            league=league, team_num=ti, columns=cols, year=year,
            league_scoring_rules={'fgm':0,'fga':0,'ftm':0,'fta':0,'threeptm':0,'reb':0,'ast':0,'stl':0,'blk':0,'turno':0,'pts':0},
            week_data=None, stat_window=stat_window
        )

        # coerce numerics
        for c in ['fgm','fga','ftm','fta','threeptm','reb','ast','stl','blk','turno','pts','fg%','ft%']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        s = df[['fgm','fga','ftm','fta','threeptm','reb','ast','stl','blk','turno','pts']].sum(numeric_only=True)

        fgm, fga = float(s.get('fgm',0)), float(s.get('fga',0))
        ftm, fta = float(s.get('ftm',0)), float(s.get('fta',0))
        fg_pct = (fgm / fga) if fga else 0.0
        ft_pct = (ftm / fta) if fta else 0.0

        cats = {
            'FG%': round(fg_pct, 4),
            'FT%': round(ft_pct, 4),
            '3PM': round(float(s.get('threeptm',0)), 2),
            'REB': round(float(s.get('reb',0)), 2),
            'AST': round(float(s.get('ast',0)), 2),
            'STL': round(float(s.get('stl',0)), 2),
            'BLK': round(float(s.get('blk',0)), 2),
            'TO' : round(float(s.get('turno',0)), 2),
            'PTS': round(float(s.get('pts',0)), 2),
        }

        records.append({
            'team_name': team.team_name,
            'abbrev': getattr(team, 'team_abbrev', None) or getattr(team, 'teamAbbrev', ''),
            'cats': cats
        })
        vol_meta.append({'FGA': fga, 'FTA': fta})

    # Build league arrays
    league_vals = {c: np.array([r['cats'][c] for r in records], dtype=float) for c in categories}

    # Invert for “better is higher”
    def better_is_higher(cat: str, arr: np.ndarray) -> np.ndarray:
        return -arr if cat == 'TO' else arr

    # z / rank / percentile / strength (min-max)
    for c in categories:
        raw = league_vals[c]
        adj = better_is_higher(c, raw)

        mu  = float(adj.mean())
        sd  = float(adj.std(ddof=0)) or 1.0
        z   = (adj - mu) / sd

        # percentile (empirical CDF); strength uses min-max on adj
        order = np.argsort(adj)              # ascending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(adj)+1)  # 1..N where higher adj => higher rank (better)
        pct   = ranks / float(len(adj))          # 0..1 (higher is better)

        mn, mx = float(adj.min()), float(adj.max())
        strength = (adj - mn) / (mx - mn) if mx > mn else np.full_like(adj, 0.5)

        # store back per-team
        for i, r in enumerate(records):
            r.setdefault('z', {})[c]         = round(float(z[i]), 3)
            r.setdefault('rank', {})[c]      = int(len(adj) - ranks[i] + 1)  # 1 = best
            r.setdefault('pct', {})[c]       = round(float(pct[i]), 3)
            r.setdefault('strength', {})[c]  = round(float(strength[i]), 3)

    # Stricter, volume-aware punts
    # Rule: A cat is “punt” if ALL are true:
    #  - z <= -0.7
    #  - bottom 3 by rank
    #  - for FG%/FT% require meaningful attempts: FGA >= league median*0.6 / FTA >= median*0.6 (avoid tiny-volume fake punts)
    # Limit to the 3 worst cats by z to avoid “everything is red” teams.
    fga_all = np.array([v['FGA'] for v in vol_meta], dtype=float)
    fta_all = np.array([v['FTA'] for v in vol_meta], dtype=float)
    fga_med = float(np.median(fga_all)) or 0.0
    fta_med = float(np.median(fta_all)) or 0.0

    for i, r in enumerate(records):
        # score candidates by z (ascending = worse first)
        candidates: List[Tuple[str, float]] = []
        for c in categories:
            z = r['z'][c]
            is_bottom3 = r['rank'][c] >= (len(records)-2)  # ranks: 1 best, N worst
            vol_ok = True
            if c == 'FG%':
                vol_ok = (vol_meta[i]['FGA'] >= 0.6 * fga_med)
            elif c == 'FT%':
                vol_ok = (vol_meta[i]['FTA'] >= 0.6 * fta_med)

            if (z <= -0.7) and is_bottom3 and vol_ok:
                candidates.append((c, z))

        # take up to 3 worst by z
        candidates.sort(key=lambda t: t[1])  # most negative first
        r['punts'] = [c for c, _ in candidates[:3]]

    return {'categories': categories, 'teams': records}
