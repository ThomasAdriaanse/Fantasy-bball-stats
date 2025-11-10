from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Dict, Any, Tuple, List

import pandas as pd

# =========================
#   Public: table schema
# =========================
def get_team_player_data_schema() -> str:
    return """
        player_id SERIAL PRIMARY KEY, 
        player_name VARCHAR(255), 
        min FLOAT, 
        fgm FLOAT, 
        fga FLOAT, 
        ftm FLOAT, 
        fta FLOAT, 
        threeptm FLOAT, 
        reb FLOAT, 
        ast FLOAT, 
        stl FLOAT, 
        blk FLOAT, 
        turno FLOAT, 
        pts FLOAT,
        fpts FLOAT,
        inj TEXT,
        games INT
    """


# =========================
#   Internal helpers
# =========================
_VALID_WINDOWS = {"projected", "total", "last_30", "last_15", "last_7"}

def _safe_boxscores(league, matchup_period: int, **kwargs) -> list:
    """ESPN sometimes throws for future weeks—normalize to []"""
    try:
        return league.box_scores(matchup_period=matchup_period, **kwargs) or []
    except Exception:
        return []

def _pick(d: dict, *keys, default=0):
    for k in keys:
        if k in d:
            return d[k]
    return default

def _r2(x) -> float:
    try:
        return round(float(x), 2)
    except Exception:
        return 0.0

def _pct(numer, denom) -> float:
    try:
        n = float(numer); d = float(denom)
        return round((n * 100.0) / d, 2) if d else 0.0
    except Exception:
        return 0.0

def _extract_avg(stats_block: dict | None) -> dict:
    if not stats_block or not isinstance(stats_block, dict):
        return {}
    if "avg" in stats_block and isinstance(stats_block["avg"], dict):
        return stats_block["avg"]
    return {}

def _date_only(dt) -> date:
    # Some ESPN dates are timezone aware-ish; you were subtracting 5–8 hours—keep that spirit
    return (dt - timedelta(hours=5)).date()


# =========================================
#   Public: team player rows (DataFrame)
# =========================================
def get_team_player_data(
    league,
    team_num: int,
    columns: List[str],
    year: int,
    league_scoring_rules: Dict[str, float],
    week_data: Dict[str, Any] | None = None,
    stat_window: str | None = None,   # "projected"|"total"|"last_30"|"last_15"|"last_7" or None
) -> pd.DataFrame:
    """
    Pull team player rows using the chosen ESPN window.

    Behavior:
      - If stat_window is provided -> STRICT mode (no fallbacks). If the window is invalid or
        has no 'avg' data, the row is 'N/A'.
      - If stat_window is None -> FRIENDLY mode with fallbacks:
            projected → total → last_30 → last_15 → last_7
    """
    team = league.teams[team_num]
    team_data = {c: [] for c in columns}

    # ensure these keys exist for output even if not listed in `columns`
    if 'games' not in team_data:
        team_data['games'] = []
    if 'games_str' not in team_data:
        team_data['games_str'] = []

    # strict vs friendly for stat window
    if stat_window is not None:
        window = (stat_window or "").strip().lower().replace("-", "_")
        candidate_keys = [f"{year}_{window}" if window in _VALID_WINDOWS else ""]
        candidate_keys = [k for k in candidate_keys if k]
    else:
        candidate_keys = [f"{year}_projected", f"{year}_total", f"{year}_last_30", f"{year}_last_15", f"{year}_last_7"]

    # establish week window once
    if week_data and week_data.get('matchup_data'):
        start_date = datetime.strptime(week_data['matchup_data']['start_date'], '%Y-%m-%d').date()
        end_date   = datetime.strptime(week_data['matchup_data']['end_date'],   '%Y-%m-%d').date()
    else:
        today_minus_8 = (datetime.today() - timedelta(hours=8)).date()
        start_date, end_date = range_of_current_week(today_minus_8)

    today_minus_8 = (datetime.today() - timedelta(hours=8)).date()

    for player in getattr(team, "roster", []):
        stats_all = getattr(player, "stats", {}) or {}

        player_avg_stats = {}
        for k in candidate_keys:
            if k in stats_all:
                player_avg_stats = _extract_avg(stats_all.get(k))
                if player_avg_stats:
                    break

        # ==== schedule-derived games (remaining / total) for this player ====
        schedule = getattr(player, "schedule", {}) or {}
        list_schedule = list(schedule.values())
        list_schedule.sort(key=lambda x: x['date'])

        games_in_week = [g for g in list_schedule if start_date <= _date_only(g['date']) <= end_date]
        games_total = len(games_in_week)
        games_remaining = sum(1 for g in games_in_week if _date_only(g['date']) >= today_minus_8)
        games_str = f"{games_remaining}/{games_total}"

        if player_avg_stats:
            # safe keys / aliases
            MIN  = _r2(_pick(player_avg_stats, "MIN", "MPG", default=0))
            FGM  = _r2(_pick(player_avg_stats, "FGM", default=0))
            FGA  = _r2(_pick(player_avg_stats, "FGA", default=0))
            FTM  = _r2(_pick(player_avg_stats, "FTM", default=0))
            FTA  = _r2(_pick(player_avg_stats, "FTA", default=0))
            TPM  = _r2(_pick(player_avg_stats, "3PM", "TPM", default=0))
            REB  = _r2(_pick(player_avg_stats, "REB", "RPG", default=0))
            AST  = _r2(_pick(player_avg_stats, "AST", "APG", default=0))
            STL  = _r2(_pick(player_avg_stats, "STL", "SPG", default=0))
            BLK  = _r2(_pick(player_avg_stats, "BLK", "BPG", default=0))
            TOs  = _r2(_pick(player_avg_stats, "TO",  "TOPG", default=0))
            PTS  = _r2(_pick(player_avg_stats, "PTS", "PPG", default=0))

            FG_PCT = _pct(FGM, FGA)
            FT_PCT = _pct(FTM, FTA)

            team_data['player_name'].append(player.name)
            team_data['min'].append(MIN)
            team_data['fgm'].append(FGM)
            team_data['fga'].append(FGA)
            team_data['fg%'].append(FG_PCT)
            team_data['ftm'].append(FTM)
            team_data['fta'].append(FTA)
            team_data['ft%'].append(FT_PCT)
            team_data['threeptm'].append(TPM)
            team_data['reb'].append(REB)
            team_data['ast'].append(AST)
            team_data['stl'].append(STL)
            team_data['blk'].append(BLK)
            team_data['turno'].append(TOs)
            team_data['pts'].append(PTS)
            team_data['inj'].append(getattr(player, "injuryStatus", None))

            fpts = round(
                FGM * league_scoring_rules.get('fgm', 0) +
                FGA * league_scoring_rules.get('fga', 0) +
                FTM * league_scoring_rules.get('ftm', 0) +
                FTA * league_scoring_rules.get('fta', 0) +
                TPM * league_scoring_rules.get('threeptm', 0) +
                REB * league_scoring_rules.get('reb', 0) +
                AST * league_scoring_rules.get('ast', 0) +
                STL * league_scoring_rules.get('stl', 0) +
                BLK * league_scoring_rules.get('blk', 0) +
                TOs * league_scoring_rules.get('turno', 0) +
                PTS * league_scoring_rules.get('pts', 0),
                2
            )
            team_data['fpts'].append(fpts)

            # games fields
            team_data['games'].append(games_remaining)
            team_data['games_str'].append(games_str)

        else:
            # Strict mode or no usable data -> N/A row
            team_data['player_name'].append(player.name)
            for k in ['min','fgm','fga','fg%','ftm','fta','ft%','threeptm','reb','ast','stl','blk','turno','pts','fpts']:
                team_data[k].append('N/A')
            team_data['inj'].append(getattr(player, "injuryStatus", None))

            # still record games for UI consistency
            team_data['games'].append(games_remaining)
            team_data['games_str'].append(games_str)

    return pd.DataFrame(team_data)


# =========================================
#   Public: points graph (cumulative)
# =========================================
def get_compare_graph(league, team1_index, team1_player_data, team2_index, team2_player_data, year, week_data=None) -> pd.DataFrame:
    team1 = league.teams[team1_index]
    team2 = league.teams[team2_index]

    # Week window
    if week_data and week_data.get('matchup_data'):
        start_date = datetime.strptime(week_data['matchup_data']['start_date'], '%Y-%m-%d').date()
        end_date   = datetime.strptime(week_data['matchup_data']['end_date'],   '%Y-%m-%d').date()
        selected_matchup_period = int(week_data['selected_week'])
    else:
        today_minus_8 = (datetime.today() - timedelta(hours=8)).date()
        start_date, end_date = range_of_current_week(today_minus_8)
        selected_matchup_period = league.currentMatchupPeriod

    dates = pd.date_range(start=start_date, end=end_date + timedelta(hours=24)).date
    dates_dict = {d: i for i, d in enumerate(dates)}

    # Predicted per day (team 1/2)
    pv1 = [0.0] * len(dates); pv1p = [0.0] * len(dates)
    pv2 = [0.0] * len(dates); pv2p = [0.0] * len(dates)

    def _accum_for_team(team, rows_df, out_all, out_from_present):
        for player in getattr(team, "roster", []):
            row = rows_df[rows_df['player_name'] == player.name]
            avg_fpts = 0 if row.empty else row['fpts'].values[0]
            if isinstance(avg_fpts, str):  # 'N/A'
                continue
            if not row.empty and row['inj'].values[0] == 'OUT':
                continue

            sched = list((getattr(player, "schedule", {}) or {}).values())
            sched.sort(key=lambda x: x['date'])
            for game in sched:
                gd = _date_only(game['date'])
                idx = dates_dict.get(gd)
                if idx is None: 
                    continue
                out_all[idx] += float(avg_fpts)
                if gd >= (datetime.today() - timedelta(hours=8)).date():
                    out_from_present[idx] += float(avg_fpts)

    _accum_for_team(team1, team1_player_data, pv1, pv1p)
    _accum_for_team(team2, team2_player_data, pv2, pv2p)

    # Fill in real points for past days (don’t double count today’s partial)
    idx1, side1 = get_team_boxscore_number(league, team1, selected_matchup_period)
    idx2, side2 = get_team_boxscore_number(league, team2, selected_matchup_period)

    # Build scoring_periods for each day
    if week_data and week_data.get('matchup_data'):
        scoring_periods = week_data['matchup_data']['scoring_periods']
    else:
        scoring_periods = []
        for i, _d in enumerate(dates):
            scoring_periods.append(i + (selected_matchup_period - 1) * 7)

    # lists to swap in past actuals
    bs1 = [0.0] * len(dates); bs2 = [0.0] * len(dates)
    today_minus_8 = (datetime.today() - timedelta(hours=8)).date()

    for i, d in enumerate(dates):
        if d >= today_minus_8:
            continue
        try:
            sp = scoring_periods[i] if i < len(scoring_periods) else i + (selected_matchup_period - 1) * 7
            boxes = _safe_boxscores(league, matchup_period=selected_matchup_period, scoring_period=sp, matchup_total=False)
            if idx1 >= 0 and idx1 < len(boxes):
                bs1[i] = float(boxes[idx1].home_score if side1 == "home" else boxes[idx1].away_score)
            if idx2 >= 0 and idx2 < len(boxes):
                bs2[i] = float(boxes[idx2].home_score if side2 == "home" else boxes[idx2].away_score)
        except Exception:
            # tolerate missing partials
            pass

    # Swap past with actuals, then cumulative sum
    for i, d in enumerate(dates):
        if d < today_minus_8:
            pv1p[i] = bs1[i]
            pv2p[i] = bs2[i]
        if i > 0:
            pv1[i]  += pv1[i - 1]
            pv2[i]  += pv2[i - 1]
            pv1p[i] += pv1p[i - 1]
            pv2p[i] += pv2p[i - 1]

    # Align arrays with your chart convention
    def _shift(a: List[float]) -> List[float]:
        a = [0.0] + a
        return a[:-1]

    pv1s  = _shift(pv1);  pv2s  = _shift(pv2)
    pv1ps = _shift(pv1p); pv2ps = _shift(pv2p)

    team1_df = pd.DataFrame({
        'date': dates,
        'predicted_fpts': pv1s,
        'predicted_fpts_from_present': pv1ps,
        'team': 'Team 1'
    })
    team2_df = pd.DataFrame({
        'date': dates,
        'predicted_fpts': pv2s,
        'predicted_fpts_from_present': pv2ps,
        'team': 'Team 2'
    })
    return pd.concat([team1_df, team2_df], ignore_index=True)


# =========================================
#   Public: categories graphs (dict of DF)
# =========================================
def get_compare_graphs_categories(league, team1_index, team1_player_data, team2_index, team2_player_data, year, week_data=None) -> Dict[str, pd.DataFrame]:
    # Week window
    if week_data and week_data.get('matchup_data'):
        start_date = datetime.strptime(week_data['matchup_data']['start_date'], '%Y-%m-%d').date()
        end_date   = datetime.strptime(week_data['matchup_data']['end_date'],   '%Y-%m-%d').date()
        selected_matchup_period = int(week_data['selected_week'])
    else:
        today_minus_8 = (datetime.today() - timedelta(hours=8)).date()
        start_date, end_date = range_of_current_week(today_minus_8)
        selected_matchup_period = league.currentMatchupPeriod

    team1 = league.teams[team1_index]
    team2 = league.teams[team2_index]
    today_minus_8 = (datetime.today() - timedelta(hours=8)).date()

    dates = pd.date_range(start=start_date, end=end_date + timedelta(hours=24)).date

    idx1, side1 = get_team_boxscore_number(league, team1, selected_matchup_period)
    idx2, side2 = get_team_boxscore_number(league, team2, selected_matchup_period)

    # scoring periods per day
    if week_data and week_data.get('matchup_data'):
        scoring_periods = week_data['matchup_data']['scoring_periods']
    else:
        scoring_periods = [i + (selected_matchup_period - 1) * 7 for i, _ in enumerate(dates)]

    # Collect actual category totals for past days (starters only to mirror projections)
    def _sum_starter_raw(lineup):
        EXCLUDE = {"BE", "IL", ""}
        totals: Dict[str, float] = {}
        for p in lineup or []:
            slot = getattr(p, "slot_position", getattr(p, "lineupSlot", ""))
            if slot in EXCLUDE:
                continue
            for k, v in (getattr(p, "points_breakdown", {}) or {}).items():
                totals[k] = totals.get(k, 0) + (v or 0)
        return totals

    def _to_cat_values(raw_totals):
        fgm = raw_totals.get("FGM", 0.0); fga = raw_totals.get("FGA", 0.0)
        ftm = raw_totals.get("FTM", 0.0); fta = raw_totals.get("FTA", 0.0)
        cat_vals = {
            "FG%": (fgm / fga) if fga else 0.0,
            "FT%": (ftm / fta) if fta else 0.0,
            "3PM": raw_totals.get("3PM", raw_totals.get("FG3M", 0.0)),
            "REB": raw_totals.get("REB", 0.0),
            "AST": raw_totals.get("AST", 0.0),
            "STL": raw_totals.get("STL", 0.0),
            "BLK": raw_totals.get("BLK", 0.0),
            "TO":  raw_totals.get("TO", 0.0),
            "PTS": raw_totals.get("PTS", 0.0),
            # keep base counters so we can recompute % curves
            "FGM": fgm, "FGA": fga, "FTM": ftm, "FTA": fta,
        }
        return {k: {"value": v} for k, v in cat_vals.items()}

    team1_box_score_list: List[Dict[str, Dict[str, float]] | int] = [0] * len(dates)
    team2_box_score_list: List[Dict[str, Dict[str, float]] | int] = [0] * len(dates)

    for i, d in enumerate(dates):
        if d >= today_minus_8:
            continue
        try:
            sp = scoring_periods[i] if i < len(scoring_periods) else i + (selected_matchup_period - 1) * 7
            boxes = _safe_boxscores(league, matchup_period=selected_matchup_period, scoring_period=sp, matchup_total=False)
            if idx1 >= 0 and idx1 < len(boxes):
                b1 = boxes[idx1]
                raw1 = _sum_starter_raw(b1.home_lineup if side1 == "home" else b1.away_lineup)
                team1_box_score_list[i] = _to_cat_values(raw1)
            if idx2 >= 0 and idx2 < len(boxes):
                b2 = boxes[idx2]
                raw2 = _sum_starter_raw(b2.home_lineup if side2 == "home" else b2.away_lineup)
                team2_box_score_list[i] = _to_cat_values(raw2)
        except Exception:
            pass

    # Projections by category (and from-present variants)
    cats = ['3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS', 'FTM', 'FTA', 'FGM', 'FGA']
    category_mapping = {
        'FG%': 'fg%',
        'FT%': 'ft%',
        '3PM': 'threeptm',
        'REB': 'reb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        'TO': 'turno',
        'PTS': 'pts',
        'FTM': 'ftm',
        'FTA': 'fta',
        'FGM': 'fgm',
        'FGA': 'fga'
    }

    pv1: Dict[str, List[float]] = {}
    pv1p: Dict[str, List[float]] = {}
    pv2: Dict[str, List[float]] = {}
    pv2p: Dict[str, List[float]] = {}

    for cat in cats:
        mapped = category_mapping[cat]
        pv1[cat], pv1p[cat], pv2[cat], pv2p[cat] = calculate_cat_predictions(
            dates, today_minus_8, team1, team2, team1_player_data, team2_player_data, mapped
        )

    combined: Dict[str, pd.DataFrame] = {}

    for cat in cats:
        # overwrite past with actuals (if present) then cumulative sum
        for i, d in enumerate(dates):
            if d < today_minus_8 and isinstance(team1_box_score_list[i], dict) and cat in team1_box_score_list[i]:
                pv1p[cat][i] = float(team1_box_score_list[i][cat]['value'])
                pv2p[cat][i] = float(team2_box_score_list[i][cat]['value'])
            if i > 0:
                pv1[cat][i]  += pv1[cat][i - 1]
                pv2[cat][i]  += pv2[cat][i - 1]
                pv1p[cat][i] += pv1p[cat][i - 1]
                pv2p[cat][i] += pv2p[cat][i - 1]

        # Align arrays with chart convention (shift by one with leading zero)
        def _shift(a: List[float]) -> List[float]:
            return [0.0] + a[:-1] if a else a

        t1_df = pd.DataFrame({
            'date': dates,
            'predicted_cat': _shift(pv1[cat]),
            'predicted_cat_from_present': _shift(pv1p[cat]),
            'team': 'Team 1',
            'category': cat
        })
        t2_df = pd.DataFrame({
            'date': dates,
            'predicted_cat': _shift(pv2[cat]),
            'predicted_cat_from_present': _shift(pv2p[cat]),
            'team': 'Team 2',
            'category': cat
        })
        combined[cat] = pd.concat([t1_df, t2_df], ignore_index=True)

    # Build FG% / FT% from cumulative counters
    def _ratio_list(num: List[float], den: List[float]) -> List[float]:
        return [(float(n) / float(d) if float(d) else 0.0) for n, d in zip(num, den)]

    fgp1  = _ratio_list(pv1['FGM'],  pv1['FGA'])
    fgp2  = _ratio_list(pv2['FGM'],  pv2['FGA'])
    fgp1p = _ratio_list(pv1p['FGM'], pv1p['FGA'])
    fgp2p = _ratio_list(pv2p['FGM'], pv2p['FGA'])

    ftp1  = _ratio_list(pv1['FTM'],  pv1['FTA'])
    ftp2  = _ratio_list(pv2['FTM'],  pv2['FTA'])
    ftp1p = _ratio_list(pv1p['FTM'], pv1p['FTA'])
    ftp2p = _ratio_list(pv2p['FTM'], pv2p['FTA'])

    # seed reasonable baselines for charts
    FG_BASE, FT_BASE = 0.47, 0.78
    for seq in (fgp1, fgp2, fgp1p, fgp2p):
        if seq:
            seq[0] = FG_BASE
    for seq in (ftp1, ftp2, ftp1p, ftp2p):
        if seq:
            seq[0] = FT_BASE

    def _mk_df(cat, t1, t1p, t2, t2p):
        t1df = pd.DataFrame({'date': dates, 'predicted_cat': t1,  'predicted_cat_from_present': t1p,
                             'team': 'Team 1', 'category': cat})
        t2df = pd.DataFrame({'date': dates, 'predicted_cat': t2,  'predicted_cat_from_present': t2p,
                             'team': 'Team 2', 'category': cat})
        return pd.concat([t1df, t2df], ignore_index=True)

    fg_df = _mk_df('FG%', fgp1, fgp1p, fgp2, fgp2p)
    ft_df = _mk_df('FT%', ftp1, ftp1p, ftp2, ftp2p)

    # Remove helper counters and return with % first
    for k in ('FTM', 'FTA', 'FGM', 'FGA'):
        combined.pop(k, None)

    ordered = {'FG%': fg_df, 'FT%': ft_df}
    for k in ['3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS']:
        if k in combined:
            ordered[k] = combined[k]
    return ordered


# =========================================
#   Public: category projections helper
# =========================================
def calculate_cat_predictions(dates, today_minus_8, team1, team2, team1_player_data, team2_player_data, mapped_cat):
    dates_dict = {d: i for i, d in enumerate(dates)}
    p1  = [0.0] * len(dates)
    p1p = [0.0] * len(dates)
    p2  = [0.0] * len(dates)
    p2p = [0.0] * len(dates)

    def _accum(team, df, out_all, out_from_present):
        for player in getattr(team, "roster", []):
            row = df[df['player_name'] == player.name]
            avg_stat = 0 if row.empty else row[mapped_cat].values[0]
            if isinstance(avg_stat, str):
                continue
            if not row.empty and row['inj'].values[0] == 'OUT':
                continue

            sched = list((getattr(player, "schedule", {}) or {}).values())
            sched.sort(key=lambda x: x['date'])
            for game in sched:
                gd = _date_only(game['date'])
                idx = dates_dict.get(gd)
                if idx is None: 
                    continue
                out_all[idx] += float(avg_stat)
                if gd >= today_minus_8:
                    out_from_present[idx] += float(avg_stat)

    _accum(team1, team1_player_data, p1, p1p)
    _accum(team2, team2_player_data, p2, p2p)

    return (
        [round(v, 2) for v in p1],
        [round(v, 2) for v in p1p],
        [round(v, 2) for v in p2],
        [round(v, 2) for v in p2p],
    )


# =========================================
#   Public: ESPN helpers
# =========================================
def get_team_boxscore_number(league, team, matchup_period=None) -> Tuple[int, str]:
    """
    Return (index_in_boxscores, 'home'|'away').
    (-1, 'home') when not found (safe default).
    """
    boxes = _safe_boxscores(league, matchup_period=matchup_period, matchup_total=False)
    for i, bx in enumerate(boxes):
        if team == getattr(bx, "home_team", None):
            return i, "home"
        if team == getattr(bx, "away_team", None):
            return i, "away"
    return -1, "home"


def range_of_current_week(day: date) -> Tuple[date, date]:
    start = day - timedelta(days=day.weekday())
    end   = start + timedelta(days=6)
    return start, end


def get_matchup_periods(league, current_matchup_period: int):
    # ESPN stores keys as strings
    return league.settings.matchup_periods[str(current_matchup_period)]
