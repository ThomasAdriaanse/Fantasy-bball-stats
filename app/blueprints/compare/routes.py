from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague
from datetime import timedelta
import time
import json

from ...services.compare_presenter import build_snapshot_rows, build_odds_rows
from ...services.espn_service import matchup_dates
from ...services.z_score_calculations import raw_to_zscore
from ...services.percent_of_win_calculations import raw_to_percent_of_win

# use your existing compare modules in project root
import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd
import compare_page.team_cat_averages as tca

bp = Blueprint("compare", __name__)


@bp.get("/select_teams_page")
def select_teams_page():
    # league_details captured by before_request already (supports ?info= and explicit params)
    league_details = session.get('league_details') or {}
    if not league_details.get('league_id') or not league_details.get('year'):
        return redirect(url_for('main.entry_page', error_message="Enter your league first."))

    try:
        if league_details.get('espn_s2') and league_details.get('swid'):
            league = League(
                league_id=league_details['league_id'],
                year=league_details['year'],
                espn_s2=league_details['espn_s2'],
                swid=league_details['swid']
            )
        else:
            league = League(
                league_id=league_details['league_id'],
                year=league_details['year']
            )
    except (ESPNUnknownError, ESPNInvalidLeague):
        return redirect(url_for('main.entry_page', error_message="Invalid league entered."))
    except ESPNAccessDenied:
        return redirect(url_for('main.entry_page', error_message="That is a private league which needs ESPN S2 and SWID."))
    except Exception as e:
        return redirect(url_for('main.entry_page', error_message=str(e)))

    year = league_details['year']
    matchup_date_data = matchup_dates(league, year)
    teams_list = [team.team_name for team in league.teams]
    form_data = session.pop('form_data', {})

    return render_template(
        'select_teams_page.html',
        info_list=teams_list,
        league_id=league_details['league_id'],
        year=league_details['year'],
        espn_s2=league_details.get('espn_s2'),
        swid=league_details.get('swid'),
        form_data=form_data,
        scoring_type=league.settings.scoring_type,
        matchup_data_dict=matchup_date_data,
        current_matchup=league.currentMatchupPeriod
    )

@bp.post("/compare_page")
def compare_page():
    start_time = time.time()
    # scoring inputs
    try:
        fgm = int(request.form.get('fgm', 2)) 
        fga = int(request.form.get('fga', -1))
        ftm = int(request.form.get('ftm', 1))
        fta = int(request.form.get('fta', -1))
        threeptm = int(request.form.get('threeptm', 1))
        reb = int(request.form.get('reb', 1))
        ast = int(request.form.get('ast', 2))
        stl = int(request.form.get('stl', 4))
        blk = int(request.form.get('blk', 4))
        turno = int(request.form.get('turno', -2))
        pts = int(request.form.get('pts', 1))
    except (ValueError, TypeError):
        flash("Invalid input. Please ensure all stats are numbers.")
        # preserve form
        info_list = [
            request.form.get('league_id'),
            request.form.get('year'),
            request.form.get('espn_s2'),
            request.form.get('swid'),
        ]
        info_string = ','.join(filter(None, info_list))
        session['form_data'] = request.form.to_dict()
        return redirect(url_for(
            'compare.select_teams_page',
            info=info_string,
            scoring_type=request.form.get('scoring_type')
        ))

    raw_window = (request.form.get('stat_window') or 'projected').strip().lower().replace('-', '_')
    VALID_WINDOWS = {'projected', 'total', 'last_30', 'last_15', 'last_7'}
    stat_window = raw_window if raw_window in VALID_WINDOWS else 'projected'

    scoring_rules = {
        'fgm': fgm, 'fga': fga, 'ftm': ftm, 'fta': fta,
        'threeptm': threeptm, 'reb': reb, 'ast': ast, 'stl': stl,
        'blk': blk, 'turno': turno, 'pts': pts
    }

    my_team_name        = request.form.get('myTeam')
    opponents_team_name = request.form.get('opponentsTeam')
    league_id           = request.form.get('league_id')
    year                = int(request.form.get('year'))
    espn_s2             = request.form.get('espn_s2')
    swid                = request.form.get('swid')
    scoring_type        = request.form.get('scoring_type')
    week_num            = int(request.form.get('week_num'))

    session['league_details'] = {
        'league_id': league_id,
        'year': int(year),
        'espn_s2': espn_s2,
        'swid': swid,
    }

    try:
        league = (
            League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
            if espn_s2 and swid
            else League(league_id=league_id, year=year)
        )
        matchup_data_dict = matchup_dates(league, year)
        selected_week_key = f'matchup_{week_num}'
        week_data = {
            "selected_week": week_num,
            "current_week": league.currentMatchupPeriod,
            "matchup_data": matchup_data_dict.get(selected_week_key, {}),
        }
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied) as e:
        error_message = "Error accessing ESPN league. Please check your league ID and credentials."
        if isinstance(e, ESPNAccessDenied):
            error_message = "This is a private league. Please provide ESPN S2 and SWID credentials."
        return redirect(url_for('main.entry_page', error_message=error_message))
    except Exception as e:
        return redirect(url_for('main.entry_page', error_message=str(e)))

    # locate teams
    team1_index = next((i for i, t in enumerate(league.teams) if t.team_name == my_team_name), -1)
    team2_index = next((i for i, t in enumerate(league.teams) if t.team_name == opponents_team_name), -1)
    if team1_index == -1:
        return redirect(url_for('main.entry_page', error_message="Team 1 not found."))
    if team2_index == -1:
        return redirect(url_for('main.entry_page', error_message="Team 2 not found."))

    session['team1_index'] = team1_index
    session['team2_index'] = team2_index

    cols = [
        'player_name', 'min', 'fgm', 'fga', 'fg%', 'ftm', 'fta', 'ft%',
        'threeptm', 'reb', 'ast', 'stl', 'blk', 'turno', 'pts', 'inj',
        'fpts', 'games',
    ]

    # ---------- POINTS SCORING ----------
    if scoring_type == "H2H_POINTS":
        t1 = cpd.get_team_player_data(league, team1_index, cols, year, scoring_rules, week_data, stat_window=stat_window)
        t2 = cpd.get_team_player_data(league, team2_index, cols, year, scoring_rules, week_data, stat_window=stat_window)

        team_cols = ['team_avg_fpts', 'team_expected_points', 'team_chance_of_winning',
                     'team_name', 'team_current_points']
        d1, d2 = tsd.get_team_stats(
            league, team1_index, t1,
            team2_index, t2,
            team_cols, scoring_rules, year, week_data
        )

        combined_df = cpd.get_compare_graph(league, team1_index, t1, team2_index, t2, year, week_data)
        combined_json = combined_df.to_json(orient='records')

        return render_template(
            'compare_page.html',
            data_team_players_1=t1.to_dict('records'),
            data_team_players_2=t2.to_dict('records'),
            data_team_stats_1=d1.to_dict('records'),
            data_team_stats_2=d2.to_dict('records'),
            combined_json=combined_json,
            scoring_type="H2H_POINTS",
            week_data=week_data,
            stat_window=stat_window,
        )

    # ---------- CATEGORY SCORING ----------
    elif scoring_type in ["H2H_CATEGORY", "H2H_MOST_CATEGORIES"]:
        t1 = cpd.get_team_player_data(league, team1_index, cols, year, scoring_rules, week_data, stat_window=stat_window)
        t2 = cpd.get_team_player_data(league, team2_index, cols, year, scoring_rules, week_data, stat_window=stat_window)

        (d1, d2, win1, win2, cur1, cur2) = tsd.get_team_stats_categories(
            league, team1_index, t1, team2_index, t2, scoring_rules, year, week_data
        )

        combined_dfs = cpd.get_compare_graphs_categories(league, team1_index, t1, team2_index, t2, year, week_data)
        combined_dicts = {cat: df.to_dict('records') for cat, df in combined_dfs.items()}

        # Build a small map of end-of-week expected percentages from the fixed math
        fg_t_exp, fg_o_exp, _, _ = tsd.get_cat_stats('fg%', 'FGM', t1, t2, cur1, cur2)
        ft_t_exp, ft_o_exp, _, _ = tsd.get_cat_stats('ft%', 'FTM', t1, t2, cur1, cur2)
        expected_pct_map = {
            'FG%': {'t1': fg_t_exp, 't2': fg_o_exp},
            'FT%': {'t1': ft_t_exp, 't2': ft_o_exp},
        }

        snapshot_rows = build_snapshot_rows(cur1, cur2)
        odds_rows = build_odds_rows(
            win1.to_dict('records'),
            combined_dicts,
            team1_current_stats=cur1,
            team2_current_stats=cur2,
            data_team_players_1=t1.to_dict('records'),
            data_team_players_2=t2.to_dict('records')
        )

        # ---------- per-player z-scores ONLY (no % of win) ----------
        import math

        def _to_float_or_nan(v):
            """
            Convert to float, but keep NaN-like values as NaN
            instead of silently turning them into 0.0.
            """
            if v is None:
                return float("nan")
            try:
                s = str(v).strip().lower()
                if s == "" or s in ("nan", "none", "null"):
                    return float("nan")
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        def _attach_metrics(df):
            """
            Attach to each row dict:
              - per-player z-scores for each cat (FG%, FT%, 3PM, REB, AST, STL, BLK, TO, PTS)
              - clean FG% / FT% decimals for display
            Returns: list of row dicts.
            """
            rows = df.to_dict('records')

            for r in rows:
                # Raw numeric fields as floats or NaN
                fgm  = _to_float_or_nan(r.get("fgm"))
                fga  = _to_float_or_nan(r.get("fga"))
                ftm  = _to_float_or_nan(r.get("ftm"))
                fta  = _to_float_or_nan(r.get("fta"))
                pts  = _to_float_or_nan(r.get("pts"))
                fg3m = _to_float_or_nan(r.get("threeptm"))
                reb  = _to_float_or_nan(r.get("reb"))
                ast  = _to_float_or_nan(r.get("ast"))
                stl  = _to_float_or_nan(r.get("stl"))
                blk  = _to_float_or_nan(r.get("blk"))
                tov  = _to_float_or_nan(r.get("turno"))

                # ----- FG% / FT% as proper decimals (0.48, 0.82, etc.) -----
                fg_pct_raw = _to_float_or_nan(r.get("fg%"))
                ft_pct_raw = _to_float_or_nan(r.get("ft%"))

                # If ESPN/your pipeline gives 48.5 instead of 0.485, normalize
                if not math.isnan(fg_pct_raw) and fg_pct_raw > 1.5:
                    fg_pct = fg_pct_raw / 100.0
                else:
                    fg_pct = fg_pct_raw

                if not math.isnan(ft_pct_raw) and ft_pct_raw > 1.5:
                    ft_pct = ft_pct_raw / 100.0
                else:
                    ft_pct = ft_pct_raw

                # Fallback: derive from makes/attempts if percentage missing
                if math.isnan(fg_pct) and not math.isnan(fgm) and not math.isnan(fga) and fga > 0:
                    fg_pct = fgm / max(fga, 1e-9)
                if math.isnan(ft_pct) and not math.isnan(ftm) and not math.isnan(fta) and fta > 0:
                    ft_pct = ftm / max(fta, 1e-9)

                # ---- Z-scores (9-cat) ----
                avg_raw_for_z = {
                    "PTS":  pts,
                    "FG3M": fg3m,
                    "REB":  reb,
                    "AST":  ast,
                    "STL":  stl,
                    "BLK":  blk,
                    "TOV":  tov,
                    "FGM":  fgm,
                    "FGA":  fga,
                    "FTM":  ftm,
                    "FTA":  fta,
                }
                z = raw_to_zscore(avg_raw_for_z) or {}

                # Keep NaN as NaN; default to NaN if missing
                r["z_pts"]      = z.get("Z_PTS",    float("nan"))
                r["z_threeptm"] = z.get("Z_FG3M",   float("nan"))
                r["z_reb"]      = z.get("Z_REB",    float("nan"))
                r["z_ast"]      = z.get("Z_AST",    float("nan"))
                r["z_stl"]      = z.get("Z_STL",    float("nan"))
                r["z_blk"]      = z.get("Z_BLK",    float("nan"))
                r["z_turno"]    = z.get("Z_TOV",    float("nan"))
                r["z_fg"]   = z.get("Z_FG", float("nan"))
                r["z_ft"]   = z.get("Z_FT", float("nan"))

                # clean FG% / FT% decimals for display (these may be NaN)
                r["fg"] = fg_pct
                r["ft"] = ft_pct

            return rows

        data_team_players_1 = _attach_metrics(t1)
        data_team_players_2 = _attach_metrics(t2)

        # ===== total games (remaining / total) per team =====
        def _sum_games(df):
            rem = 0
            tot = 0
            if 'games_str' in df.columns:
                for s in df['games_str'].fillna(''):
                    if isinstance(s, str) and '/' in s:
                        left, right = s.split('/', 1)
                        try:
                            rem += int(left)
                            tot += int(right)
                        except Exception:
                            continue
            if (rem == 0 and tot == 0) and ('games' in df.columns):
                val = int(df['games'].fillna(0).sum())
                rem = val
                tot = val
            return rem, tot

        team1_games_remaining, team1_games_total = _sum_games(t1)
        team2_games_remaining, team2_games_total = _sum_games(t2)

        return render_template(
            "compare_page_cat.html",
            data_team_players_1=data_team_players_1,
            data_team_players_2=data_team_players_2,
            data_team_stats_1=d1.to_dict('records'),
            data_team_stats_2=d2.to_dict('records'),
            team1_win_pct_data=win1.to_dict('records'),
            team2_win_pct_data=win2.to_dict('records'),
            team1_current_stats=cur1,
            team2_current_stats=cur2,
            combined_jsons=combined_dicts,
            scoring_type=scoring_type,
            week_data=week_data,
            stat_window=stat_window,
            expected_pct_map=expected_pct_map,
            snapshot_rows=snapshot_rows,
            odds_rows=odds_rows,
            team1_games_remaining=team1_games_remaining,
            team1_games_total=team1_games_total,
            team2_games_remaining=team2_games_remaining,
            team2_games_total=team2_games_total,
            league_id=league_id,
            year=year,
            espn_s2=espn_s2,
            swid=swid,
        )

    return redirect(url_for('main.entry_page', error_message=f"Unsupported scoring type: {scoring_type}"))
