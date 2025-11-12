from __future__ import annotations

import time
from typing import Dict, Tuple, Any, List

import pandas as pd
from scipy.stats import norm
import math

# stays as-is since you kept the package at project root
import compare_page.compare_page_data as cpd


def _safe_boxscores(league, matchup_period: int, **kwargs) -> list:
    """
    Fetch box scores once and return a list; safely handle ESPN quirks for future weeks.
    """
    try:
        return league.box_scores(matchup_period=matchup_period, **kwargs) or []
    except Exception:
        # For future weeks (or ESPN API quirks), ESPN can throw; normalize to empty list.
        return []


def _locate_teams_in_boxscores(league, matchup_period: int, team_idx: int, opp_idx: int) -> Tuple[int, str, int, str]:
    """
    Find each team’s index in box scores and whether they are home/away.
    Returns: (team_boxscore_num, team_home_or_away, opp_boxscore_num, opp_home_or_away)
    """
    boxes = _safe_boxscores(league, matchup_period=matchup_period)
    team_boxscore_num = -1
    opp_boxscore_num = -1
    team_home_or_away = "temp"
    opp_home_or_away = "temp"

    for i, bx in enumerate(boxes):
        if league.teams[team_idx] == getattr(bx, "home_team", None):
            team_boxscore_num = i
            team_home_or_away = "home"
        elif league.teams[team_idx] == getattr(bx, "away_team", None):
            team_boxscore_num = i
            team_home_or_away = "away"

        if league.teams[opp_idx] == getattr(bx, "home_team", None):
            opp_boxscore_num = i
            opp_home_or_away = "home"
        elif league.teams[opp_idx] == getattr(bx, "away_team", None):
            opp_boxscore_num = i
            opp_home_or_away = "away"

    return team_boxscore_num, team_home_or_away, opp_boxscore_num, opp_home_or_away


def _team_avg_fpts(league, team, year: int, rules: Dict[str, float]) -> float:
    """
    Compute a team’s average fantasy points over roster using season averages.
    Guards against empty/partial data.
    """
    year_key = f"{year}_total"
    total = 0.0
    count = 0

    for p in getattr(team, "roster", []):
        stats = getattr(p, "stats", {}) or {}
        year_blob = stats.get(year_key, {})
        avg = year_blob.get("avg") if isinstance(year_blob, dict) else None
        if not avg:
            continue

        # Some ESPN fields are 3PM/TO etc.
        try:
            fpts = (
                avg.get("FGM", 0) * rules["fgm"] +
                avg.get("FGA", 0) * rules["fga"] +
                avg.get("FTM", 0) * rules["ftm"] +
                avg.get("FTA", 0) * rules["fta"] +
                avg.get("3PM", 0) * rules["threeptm"] +
                avg.get("REB", 0) * rules["reb"] +
                avg.get("AST", 0) * rules["ast"] +
                avg.get("STL", 0) * rules["stl"] +
                avg.get("BLK", 0) * rules["blk"] +
                avg.get("TO",  0) * rules["turno"] +
                avg.get("PTS", 0) * rules["pts"]
            )
        except KeyError:
            # If a rule key is missing, treat it as zero weight.
            fpts = 0.0
        total += float(fpts)
        count += 1

    if count == 0:
        return 0.0
    return round(total / count, 2)


def get_team_stats(
    league,
    team_num: int,
    team_player_data: pd.DataFrame,
    opponent_num: int,
    opponent_player_data: pd.DataFrame,
    columns: List[str],
    league_scoring_rules: Dict[str, float],
    year: int,
    week_data: Dict[str, Any] | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Points scoring view: expected totals + win probability.
    Returns (team_df, opponent_df).
    """
    selected_matchup_period = week_data['selected_week'] if week_data else league.currentMatchupPeriod

    # Find teams in this week’s boxscores once
    team_boxscore_num, team_side, opp_boxscore_num, opp_side = _locate_teams_in_boxscores(
        league, selected_matchup_period, team_num, opponent_num
    )

    team = league.teams[team_num]
    opponent = league.teams[opponent_num]
    team_data = {c: [] for c in columns}
    opponent_data = {c: [] for c in columns}

    # Team average FPTS (season avg)
    team_data['team_avg_fpts'].append(_team_avg_fpts(league, team, year, league_scoring_rules))
    opponent_data['team_avg_fpts'].append(_team_avg_fpts(league, opponent, year, league_scoring_rules))

    # Projected points left this week (exclude OUT players, exclude N/A rows)
    def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
        df2 = df[(df['inj'] != 'OUT') & (df['fpts'] != 'N/A')].copy()
        df2['fpts'] = pd.to_numeric(df2['fpts'], errors='coerce')
        df2['games'] = pd.to_numeric(df2['games'], errors='coerce')
        return df2.dropna(subset=['fpts', 'games'])

    t_df = _valid_rows(team_player_data)
    o_df = _valid_rows(opponent_player_data)

    team_expected_points_remaining = float((t_df['fpts'] * t_df['games']).sum())
    opp_expected_points_remaining  = float((o_df['fpts'] * o_df['games']).sum())

    # Standard deviation assumptions
    player_std_dev = 0.40
    t_std = (t_df['fpts'] * player_std_dev * t_df['games']).pow(2).sum() ** 0.5
    o_std = (o_df['fpts'] * player_std_dev * o_df['games']).pow(2).sum() ** 0.5

    # Current points (subtract today’s partial period so projections don’t double-count)
    current_period = getattr(league, "scoringPeriodId", None)
    week_boxes_total = _safe_boxscores(league, matchup_period=selected_matchup_period)
    if current_period is not None:
        week_boxes_current = _safe_boxscores(league, matchup_period=selected_matchup_period,
                                             scoring_period=current_period, matchup_total=False)
    else:
        week_boxes_current = []

    def _current_for(index: int, side: str) -> float:
        if index < 0 or index >= len(week_boxes_total):
            return 0.0
        try:
            if side == "home":
                total = float(getattr(week_boxes_total[index], "home_score", 0) or 0)
                if index < len(week_boxes_current):
                    total -= float(getattr(week_boxes_current[index], "home_score", 0) or 0)
            else:
                total = float(getattr(week_boxes_total[index], "away_score", 0) or 0)
                if index < len(week_boxes_current):
                    total -= float(getattr(week_boxes_current[index], "away_score", 0) or 0)
            return total
        except Exception:
            return 0.0

    team_current_points = _current_for(team_boxscore_num, team_side)
    opp_current_points  = _current_for(opp_boxscore_num, opp_side)

    team_total_expected = team_expected_points_remaining + team_current_points
    opp_total_expected  = opp_expected_points_remaining + opp_current_points

    # Win probability
    diff = team_total_expected - opp_total_expected
    denom = (t_std ** 2 + o_std ** 2) ** 0.5
    if denom > 0:
        p_team = float(norm.cdf(diff / denom))
    else:
        # No variance; deterministic outcome
        p_team = 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5)

    # Fill rows
    team_data['team_expected_points'].append(round(team_total_expected, 2))
    team_data['team_chance_of_winning'].append(round(p_team * 100, 2))
    team_data['team_name'].append(team.team_name)
    team_data['team_current_points'].append(round(team_current_points, 2))

    opponent_data['team_expected_points'].append(round(opp_total_expected, 2))
    opponent_data['team_chance_of_winning'].append(round((1 - p_team) * 100, 2))
    opponent_data['team_name'].append(opponent.team_name)
    opponent_data['team_current_points'].append(round(opp_current_points, 2))

    return pd.DataFrame(team_data), pd.DataFrame(opponent_data)


def get_team_stats_categories(
    league,
    team_num: int,
    team_player_data: pd.DataFrame,
    opponent_num: int,
    opponent_player_data: pd.DataFrame,
    league_scoring_rules: Dict[str, float],
    year: int,
    week_data: Dict[str, Any] | None = None
):
    """
    Categories view: expected category totals and win% per category.
    Returns:
      (team_df, opponent_df, team_pct_df, opponent_pct_df, team_current_stats, opponent_current_stats)
    """
    team = league.teams[team_num]
    opponent = league.teams[opponent_num]

    selected_matchup_period = week_data['selected_week'] if week_data else league.currentMatchupPeriod

    # Locate box scores and current category totals (if available)
    try:
        team_boxscore_num, team_side, opp_boxscore_num, opp_side = _locate_teams_in_boxscores(
            league, selected_matchup_period, team_num, opponent_num
        )
        boxes = _safe_boxscores(league, matchup_period=selected_matchup_period)

        def _current_stats(index: int, side: str) -> Dict[str, Any]:
            if index < 0 or index >= len(boxes):
                return {}
            if side == "home":
                return getattr(boxes[index], "home_stats", {}) or {}
            return getattr(boxes[index], "away_stats", {}) or {}

        team_current_stats = _current_stats(team_boxscore_num, team_side)
        opponent_current_stats = _current_stats(opp_boxscore_num, opp_side)
    except Exception:
        team_current_stats = {}
        opponent_current_stats = {}

    # Your category map
    cats      = ['fg%', 'ft%', 'threeptm', 'reb', 'ast', 'stl', 'blk', 'turno', 'pts']
    espn_cats = ['FG%', 'FT%', '3PM',     'REB', 'AST', 'STL', 'BLK', 'TO',    'PTS']

    team_expected, opp_expected = {}, {}
    team_win_pct, opp_win_pct = {}, {}

    for cat, espn_cat in zip(cats, espn_cats):
        texp, opexp, twin, owin = get_cat_stats(
            cat, espn_cat,
            team_player_data, opponent_player_data,
            team_current_stats, opponent_current_stats
        )
        team_expected[cat] = texp
        opp_expected[cat] = opexp
        team_win_pct[cat] = twin
        opp_win_pct[cat] = owin

    # Meta for header compatibility
    team_expected['team_name'] = team.team_name
    opp_expected['team_name'] = opponent.team_name
    team_expected['team_current_points'] = ''   # not meaningful on categories page
    opp_expected['team_current_points'] = ''

    df_team = pd.DataFrame([team_expected])
    df_opp  = pd.DataFrame([opp_expected])
    df_team_pct = pd.DataFrame([team_win_pct])
    df_opp_pct  = pd.DataFrame([opp_win_pct])

    return df_team, df_opp, df_team_pct, df_opp_pct, team_current_stats, opponent_current_stats


def get_cat_stats(
    cat: str,
    espn_cat: str,
    team_player_data: pd.DataFrame,
    opponent_player_data: pd.DataFrame,
    team_current_stats: Dict[str, Any],
    opponent_current_stats: Dict[str, Any]
) -> Tuple[float, float, float, float]:
    """
    Compute team/opponent expected category totals and win % for one category.
    Returns: (team_expected, opponent_expected, team_win_pct, opponent_win_pct)
    """
    def _clean(df: pd.DataFrame, stat: str) -> pd.DataFrame:
        out = df[(df['inj'] != 'OUT') & (df[stat] != 'N/A')].copy()
        out[stat] = pd.to_numeric(out[stat], errors='coerce')
        out['games'] = pd.to_numeric(out['games'], errors='coerce')
        return out.dropna(subset=[stat, 'games'])

    t_df = _clean(team_player_data, cat)
    o_df = _clean(opponent_player_data, cat)

    if cat in ('fg%', 'ft%'):
        made = 'fgm' if cat == 'fg%' else 'ftm'
        att  = 'fga' if cat == 'fg%' else 'fta'

        def _ensure_num(df, cols):
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df.dropna(subset=cols)

        # Filter and ensure numeric for per-player projected rates + games
        t_df = _clean(team_player_data, made)
        t_df = _ensure_num(t_df, [made, att, 'games'])
        o_df = _clean(opponent_player_data, made)
        o_df = _ensure_num(o_df, [made, att, 'games'])

        # Project *future* makes/attempts (rate * remaining games)
        t_exp_makes = float((t_df[made] * t_df['games']).sum())
        t_exp_atts  = float((t_df[att]  * t_df['games']).sum())
        o_exp_makes = float((o_df[made] * o_df['games']).sum())
        o_exp_atts  = float((o_df[att]  * o_df['games']).sum())

        # 🔧 FIX: current cumulative box totals use UPPERCASE keys in team_current_stats
        made_key = made.upper()  # 'FGM' or 'FTM'
        att_key  = att.upper()   # 'FGA' or 'FTA'

        t_cur_m = float((team_current_stats.get(made_key, {}) or {}).get('value', 0) or 0)
        t_cur_a = float((team_current_stats.get(att_key,  {}) or {}).get('value', 0) or 0)
        o_cur_m = float((opponent_current_stats.get(made_key, {}) or {}).get('value', 0) or 0)
        o_cur_a = float((opponent_current_stats.get(att_key,  {}) or {}).get('value', 0) or 0)

        # End-of-week totals (projected)
        t_tot_m = t_cur_m + t_exp_makes
        t_tot_a = t_cur_a + t_exp_atts
        o_tot_m = o_cur_m + o_exp_makes
        o_tot_a = o_cur_a + o_exp_atts

        # Expected proportions (0..1). Display as 0..100 later.
        t_p = (t_tot_m / t_tot_a) if t_tot_a > 0 else None
        o_p = (o_tot_m / o_tot_a) if o_tot_a > 0 else None

        if t_p is None and o_p is None:
            t_expected, o_expected, twin = 0.0, 0.0, 50.0
        elif t_p is None:
            t_expected, o_expected, twin = 0.0, o_p * 100.0, 0.0
        elif o_p is None:
            t_expected, o_expected, twin = t_p * 100.0, 0.0, 100.0
        else:
            # Variance of difference in proportions:
            # Var(p1 - p2) ≈ p1(1-p1)/A1 + p2(1-p2)/A2
            var_diff = (t_p * (1.0 - t_p)) / max(1.0, t_tot_a) + (o_p * (1.0 - o_p)) / max(1.0, o_tot_a)
            sd = math.sqrt(max(var_diff, 1e-12))
            z = (t_p - o_p) / sd
            twin = float(norm.cdf(z) * 100.0)

            t_expected = t_p * 100.0
            o_expected = o_p * 100.0

            print("[FT%/FG% DEBUG]",
            {"t_cur": (t_cur_m, t_cur_a), "o_cur": (o_cur_m, o_cur_a),
            "t_exp": (t_exp_makes, t_exp_atts), "o_exp": (o_exp_makes, o_exp_atts),
            "t_tot": (t_tot_m, t_tot_a), "o_tot": (o_tot_m, o_tot_a)})


    else:
        # counting stats
        t_cur = float((team_current_stats.get(espn_cat, {}) or {}).get('value', 0) or 0)
        o_cur = float((opponent_current_stats.get(espn_cat, {}) or {}).get('value', 0) or 0)

        t_expected = t_cur + float((t_df[cat] * t_df['games']).sum())
        o_expected = o_cur + float((o_df[cat] * o_df['games']).sum())

        if cat == 'pts':
            std_factor = 0.25
        elif cat in ('threeptm', 'stl', 'blk'):
            std_factor = 0.60
        else:
            std_factor = 0.40

        t_var = float(((t_df[cat] * std_factor * t_df['games']) ** 2).sum())
        o_var = float(((o_df[cat] * std_factor * o_df['games']) ** 2).sum())
        denom = (t_var + o_var) ** 0.5

        # For turnovers, lower is better -> invert sign
        diff = (o_expected - t_expected) if cat == 'turno' else (t_expected - o_expected)

        if denom > 0:
            twin = float(norm.cdf(diff / denom) * 100)
        else:
            twin = 50.0 if diff == 0 else (100.0 if diff > 0 else 0.0)

    owin = 100.0 - twin
    return round(t_expected, 2), round(o_expected, 2), round(twin, 2), round(owin, 2)
