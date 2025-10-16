from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague
from datetime import timedelta
import time
import json

# use your existing compare modules in project root
import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd
import compare_page.team_cat_averages as tca

from ...services.league_session import _parse_from_req, _store
from ...services.espn_service import matchup_dates

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
        info_list = [request.form.get('league_id'), request.form.get('year'), request.form.get('espn_s2'), request.form.get('swid')]
        info_string = ','.join(filter(None, info_list))
        session['form_data'] = request.form.to_dict()
        return redirect(url_for('compare.select_teams_page', info=info_string, scoring_type=request.form.get('scoring_type')))

    raw_window = (request.form.get('stat_window') or 'projected').strip().lower().replace('-', '_')
    VALID_WINDOWS = {'projected', 'total', 'last_30', 'last_15', 'last_7'}
    stat_window = raw_window if raw_window in VALID_WINDOWS else 'projected'

    scoring_rules = {'fgm': fgm, 'fga': fga, 'ftm': ftm, 'fta': fta,
                     'threeptm': threeptm, 'reb': reb, 'ast': ast, 'stl': stl,
                     'blk': blk, 'turno': turno, 'pts': pts}

    my_team_name        = request.form.get('myTeam')
    opponents_team_name = request.form.get('opponentsTeam')
    league_id           = request.form.get('league_id')
    year                = int(request.form.get('year'))
    espn_s2             = request.form.get('espn_s2')
    swid                = request.form.get('swid')
    scoring_type        = request.form.get('scoring_type')
    week_num            = int(request.form.get('week_num'))

    _store({'league_id': league_id, 'year': year, 'espn_s2': espn_s2, 'swid': swid})

    try:
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid) if espn_s2 and swid else League(league_id=league_id, year=year)
        matchup_data_dict = matchup_dates(league, year)
        selected_week_key = f'matchup_{week_num}'
        week_data = {
            "selected_week": week_num,
            "current_week": league.currentMatchupPeriod,
            "matchup_data": matchup_data_dict.get(selected_week_key, {})
        }
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied) as e:
        error_message = "Error accessing ESPN league. Please check your league ID and credentials."
        if isinstance(e, ESPNAccessDenied):
            error_message = "This is a private league. Please provide ESPN S2 and SWID credentials."
        return redirect(url_for('main.entry_page', error_message=error_message))
    except Exception as e:
        return redirect(url_for('main.entry_page', error_message=str(e)))

    # locate teams
    team1_index = next((i for i,t in enumerate(league.teams) if t.team_name == my_team_name), -1)
    team2_index = next((i for i,t in enumerate(league.teams) if t.team_name == opponents_team_name), -1)
    if team1_index == -1: return redirect(url_for('main.entry_page', error_message="Team 1 not found."))
    if team2_index == -1: return redirect(url_for('main.entry_page', error_message="Team 2 not found."))

    cols = ['player_name','min','fgm','fga','fg%','ftm','fta','ft%','threeptm','reb','ast','stl','blk','turno','pts','inj','fpts','games']

    if scoring_type == "H2H_POINTS":
        t1 = cpd.get_team_player_data(league, team1_index, cols, year, scoring_rules, week_data, stat_window=stat_window)
        t2 = cpd.get_team_player_data(league, team2_index, cols, year, scoring_rules, week_data, stat_window=stat_window)

        team_cols = ['team_avg_fpts','team_expected_points','team_chance_of_winning','team_name','team_current_points']
        d1, d2 = tsd.get_team_stats(league, team1_index, t1, team2_index, t2, team_cols, scoring_rules, year, week_data)

        combined_df = cpd.get_compare_graph(league, team1_index, t1, team2_index, t2, year, week_data)
        combined_json = combined_df.to_json(orient='records')

        return render_template('compare_page.html',
                               data_team_players_1=t1.to_dict('records'),
                               data_team_players_2=t2.to_dict('records'),
                               data_team_stats_1=d1.to_dict('records'),
                               data_team_stats_2=d2.to_dict('records'),
                               combined_json=combined_json,
                               scoring_type="H2H_POINTS",
                               week_data=week_data,
                               stat_window=stat_window)

    elif scoring_type in ["H2H_CATEGORY", "H2H_MOST_CATEGORIES"]:
        t1 = cpd.get_team_player_data(league, team1_index, cols, year, scoring_rules, week_data, stat_window=stat_window)
        t2 = cpd.get_team_player_data(league, team2_index, cols, year, scoring_rules, week_data, stat_window=stat_window)

        (d1, d2, win1, win2, cur1, cur2) = tsd.get_team_stats_categories(
            league, team1_index, t1, team2_index, t2, scoring_rules, year, week_data
        )
        combined_dfs = cpd.get_compare_graphs_categories(league, team1_index, t1, team2_index, t2, year, week_data)
        combined_dicts = {cat: df.to_dict('records') for cat, df in combined_dfs.items()}

        return render_template('compare_page_cat.html',
                               data_team_players_1=t1.to_dict('records'),
                               data_team_players_2=t2.to_dict('records'),
                               data_team_stats_1=d1.to_dict('records'),
                               data_team_stats_2=d2.to_dict('records'),
                               team1_win_pct_data=win1.to_dict('records'),
                               team2_win_pct_data=win2.to_dict('records'),
                               team1_current_stats=cur1, team2_current_stats=cur2,
                               combined_jsons=combined_dicts,
                               scoring_type=scoring_type, week_data=week_data,
                               stat_window=stat_window)

    return redirect(url_for('main.entry_page', error_message=f"Unsupported scoring type: {scoring_type}"))
