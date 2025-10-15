import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd

def _team_category_averages(league, year, stat_window='projected'):
    """
    Returns:
      {
        'categories': ['FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS'],
        'teams': [
          {
            'team_name': '…',
            'abbrev': '…',
            'cats': {'FG%': 0.48, 'FT%': 0.79, '3PM': 10.2, 'REB': 48.7, 'AST': 23.4, 'STL': 7.9, 'BLK': 4.2, 'TO': 13.7, 'PTS': 108.3},
            'z':     {'FG%': 0.50, 'FT%': -0.12, '3PM': 0.8, ...},
            'punts': ['FT%', 'AST']   # categories the team is likely “punting”
          },
          ...
        ]
      }
    """
    import numpy as np
    import pandas as pd

    # columns your cpd.get_team_player_data expects/returns
    cols = [
        'player_name','min','fgm','fga','fg%','ftm','fta','ft%','threeptm',
        'reb','ast','stl','blk','turno','pts','inj','fpts','games'
    ]

    # Compute each team’s averages by summing player averages (your “team average” notion)
    records = []
    for ti, team in enumerate(league.teams):
        df = cpd.get_team_player_data(
            league, ti, cols, year,
            league_scoring_rules={
                'fgm':0,'fga':0,'ftm':0,'fta':0,'threeptm':0,'reb':0,'ast':0,'stl':0,'blk':0,'turno':0,'pts':0
            },
            week_data=None,
            stat_window=stat_window  # strict mode: only this ESPN window
        )

        # Coerce numerics, ignore N/A rows
        num_cols = ['fgm','fga','ftm','fta','threeptm','reb','ast','stl','blk','turno','pts']
        for c in num_cols+['fg%','ft%']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Sum player averages to get team "per-game" totals
        s = df[num_cols].sum(numeric_only=True)

        # Derive percentages from summed makes/attempts (more stable than mean of percentages)
        fgm, fga, ftm, fta = s['fgm'], s['fga'], s['ftm'], s['fta']
        fg_pct = float(fgm) / float(fga) if fga else 0.0
        ft_pct = float(ftm) / float(fta) if fta else 0.0

        cats = {
            'FG%': round(fg_pct, 4),
            'FT%': round(ft_pct, 4),
            '3PM': round(s['threeptm'], 2),
            'REB': round(s['reb'], 2),
            'AST': round(s['ast'], 2),
            'STL': round(s['stl'], 2),
            'BLK': round(s['blk'], 2),
            'TO' : round(s['turno'], 2),
            'PTS': round(s['pts'], 2),
        }

        records.append({
            'team_name': team.team_name,
            'abbrev': getattr(team, 'team_abbrev', None) or getattr(team, 'teamAbbrev', ''),
            'cats': cats
        })

    # Z-scores across league for punt detection
    categories = ['FG%','FT%','3PM','REB','AST','STL','BLK','TO','PTS']
    # Build arrays
    mat = {c: np.array([r['cats'][c] for r in records], dtype=float) for c in categories}

    # For turnovers, “better” is LOWER — invert for z so low TO => positive z
    invert_for_to = True
    for c in categories:
        arr = mat[c]
        if c == 'TO' and invert_for_to:
            arr = -arr
        mu, sd = float(np.mean(arr)), float(np.std(arr, ddof=0)) or 1.0
        zs = (arr - mu) / sd
        # write back (remember if inverted TO)
        if c == 'TO' and invert_for_to:
            # We computed z on -TO; store that z (positive means good, i.e., low TO)
            z_to_store = zs
        else:
            z_to_store = zs
        for i, r in enumerate(records):
            r.setdefault('z', {})[c] = round(float(z_to_store[i]), 3)

    # Punt heuristic:
    #   - z <= -0.6  OR  team ranks in bottom 3 for that category (using “good = high” except TO)
    for c in categories:
        values = mat[c]
        # ranks: higher is better except TO
        order = np.argsort(values)  # ascending
        if c != 'TO':
            bottom3_idx = set(order[:3])
        else:
            # for TO, higher is worse; bottom3 in “goodness” == top3 of TO values
            bottom3_idx = set(order[-3:])

        for i, r in enumerate(records):
            z = r['z'][c]
            is_bottom = i in bottom3_idx
            if z <= -0.6 or is_bottom:
                r.setdefault('punts', []).append(c)
        # Sort punts for stable display
        for r in records:
            r['punts'] = sorted(r.get('punts', []))

    return {'categories': categories, 'teams': records}
