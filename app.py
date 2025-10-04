from flask import Flask, render_template, redirect, url_for, request, flash, session, send_from_directory
import os
from dotenv import load_dotenv
from psycopg2 import extras
import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd
from scipy.stats import norm
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague
import pandas as pd
import db_utils
from nba_api.stats.endpoints import playergamelog, playercareerstats, leaguegamefinder
from nba_api.stats.static import players
import json
import time
from pprint import pprint
from datetime import datetime, timedelta

load_dotenv()
app = Flask(__name__)

app.secret_key = os.urandom(24)

# Define the get_matchup_dates function at the global level so it can be used by multiple routes
def get_matchup_dates(league):
    """Generate matchup date data for all matchup periods in the league"""
    today = datetime.today()
    current_year = today.year

    if today > datetime(current_year, 3, 30) and today < datetime(current_year, 10, 20):
        season_start = datetime(2025, 4, 13).date() - timedelta(days=league.scoringPeriodId)
    else:
        season_start = today - timedelta(days=league.scoringPeriodId)
    
    matchupperiods = league.settings.matchup_periods
    first_scoring_period = league.firstScoringPeriod

    #print(playoff_matchup_period_length)
    #print(vars(playoff_matchup_period_length))

    matchup_date_data = {}
    
    matchup_period_keys = [int(k) for k in matchupperiods.keys()]
    max_matchup_period = max(matchup_period_keys)
    
    prev_end_date = None  # Initialize prev_end_date variable
    scoring_period_multiplier = 0

    #print(matchupperiods)
    #print(matchupperiods.items())

    for matchup_period_number, matchup_periods in matchupperiods.items():
        matchup_period_number = int(matchup_period_number)
        
        matchup_length = len(matchup_periods)

        if matchup_period_number == (18 - first_scoring_period):
            matchup_length += 1

        if matchup_period_number == 1:
            start_date = season_start
            end_date = start_date + timedelta(days=6) + timedelta(days=7)*(matchup_length-1)  # First week is one day shorter

        else:
            start_date = prev_end_date + timedelta(days=1)
            end_date = start_date + timedelta(days=6)+ timedelta(days=7)*(matchup_length-1)  # Regular weeks are 7 days

        scoring_periods = [i + (matchup_period_number - 1 + scoring_period_multiplier) * 7 for i in range((end_date - start_date).days + 1)]
        
        matchup_date_data[f'matchup_{matchup_period_number}'] = {
            'matchup_period': matchup_period_number,
            'scoring_periods': scoring_periods,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }

        # we have to adjust the scoring period for past matchups with multiple weeks
        scoring_period_multiplier += matchup_length-1

        prev_end_date = end_date
    #print(matchup_date_data)
    return matchup_date_data

@app.route('/')
def entry_page():
    error_message = request.args.get('error_message', '')
    return render_template('entry.html', error_message=error_message)

@app.route('/player_stats', methods=['GET', 'POST'])
def player_stats():
    num_games = 20  # Default window
    selected_stat = 'FPTS'  # Default stat

    if request.method == 'GET':
        selected_player = request.args.get('player_name', 'Bol Bol')
        selected_stat   = request.args.get('stat', selected_stat)
    else:  # POST
        selected_player = request.form.get('player_name', 'Bol Bol')
        num_games       = int(request.form.get('num_games', num_games))
        selected_stat   = request.form.get('stat', selected_stat)

    # Build/update the JSON the chart reads
    generate_json_file(selected_player, num_games, selected_stat)

    # Players list for dropdown
    active_players = players.get_active_players()
    active_players.sort(key=lambda x: x['full_name'])

    # Expose stat options to the template (template can ignore for now)
    stat_options = [
        'FPTS','PTS','REB','AST','STL','BLK','TOV',
        'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT',
        'FTM','FTA','FT_PCT','MIN','PLUS_MINUS'
    ]

    return render_template(
        'player_stats.html',
        players=active_players,
        selected_player=selected_player,
        num_games=num_games,
        selected_stat=selected_stat,
        stat_options=stat_options
    )


# Route to serve the JSON file for player stats
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('static', filename)

def generate_json_file(player_name, num_games, stat='FPTS'):
    player_info = players.find_players_by_full_name(player_name)
    if not player_info:
        print(f"Player {player_name} not found.")
        return

    player_id = player_info[0]['id']
    print(f"Fetching ALL regular-season games for: {player_name} (ID: {player_id}), stat={stat}")

    try:
        lgf = leaguegamefinder.LeagueGameFinder(
            player_id_nullable=player_id,
            player_or_team_abbreviation="P",
            season_type_nullable="Regular Season"
        )
        all_seasons_df = lgf.get_data_frames()[0]
    except Exception as e:
        print(f"LeagueGameFinder error: {e}")
        return

    if all_seasons_df.empty:
        print("No data available for the specified player.")
        return

    # Coerce numerics
    numeric_cols = [
        "FGM","FGA","FG3M","FG3A","FTM","FTA",
        "REB","AST","STL","BLK","TOV","PTS","MIN",
        "FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"
    ]
    for c in numeric_cols:
        if c in all_seasons_df.columns:
            all_seasons_df[c] = pd.to_numeric(all_seasons_df[c], errors="coerce")

    # Fantasy points (always compute; used if stat == 'FPTS')
    league_scoring_rules = {
        'fgm': 2, 'fga': -1, 'ftm': 1, 'fta': -1,
        'threeptm': 1, 'reb': 1, 'ast': 2, 'stl': 4,
        'blk': 4, 'turno': -2, 'pts': 1
    }

    def calculate_fp_row(row):
        return (
            (row.get("FGM", 0) or 0)   * league_scoring_rules['fgm'] +
            (row.get("FGA", 0) or 0)   * league_scoring_rules['fga'] +
            (row.get("FTM", 0) or 0)   * league_scoring_rules['ftm'] +
            (row.get("FTA", 0) or 0)   * league_scoring_rules['fta'] +
            (row.get("FG3M", 0) or 0)  * league_scoring_rules['threeptm'] +
            (row.get("REB", 0) or 0)   * league_scoring_rules['reb'] +
            (row.get("AST", 0) or 0)   * league_scoring_rules['ast'] +
            (row.get("STL", 0) or 0)   * league_scoring_rules['stl'] +
            (row.get("BLK", 0) or 0)   * league_scoring_rules['blk'] +
            (row.get("TOV", 0) or 0)   * league_scoring_rules['turno'] +
            (row.get("PTS", 0) or 0)   * league_scoring_rules['pts']
        )

    all_seasons_df['FPTS'] = all_seasons_df.apply(calculate_fp_row, axis=1)

    # Dates & ordering
    all_seasons_df['Game_Date'] = pd.to_datetime(all_seasons_df['GAME_DATE'], errors="coerce")
    all_seasons_df = all_seasons_df.sort_values('Game_Date').reset_index(drop=True)
    all_seasons_df['Game_Number'] = all_seasons_df.index + 1

    # Season label helpers
    all_seasons_df['SEASON_ID'] = all_seasons_df['SEASON_ID'].astype(str)
    all_seasons_df['SEASON_START'] = all_seasons_df['SEASON_ID'].str[-4:].astype(int)
    all_seasons_df['SEASON'] = all_seasons_df['SEASON_START'].map(
        lambda y: f"{y}-{str((y+1) % 100).zfill(2)}"
    )

    # Which column to average?
    # Allow only known safe columns; fall back to FPTS if unknown
    allowed = set([
        'FPTS','PTS','REB','AST','STL','BLK','TOV',
        'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT',
        'FTM','FTA','FT_PCT','MIN','PLUS_MINUS'
    ])
    value_col = stat if stat in allowed else 'FPTS'

    # Centered moving average of the chosen stat
    def centered_average(idx, half_window):
        start = max(0, idx - half_window)
        end   = min(len(all_seasons_df), idx + half_window + 1)
        return all_seasons_df[value_col].iloc[start:end].mean()

    all_seasons_df['Centered_Avg'] = all_seasons_df.index.map(
        lambda i: centered_average(i, num_games)
    )

    # Prepare export (keep original key names for your current front-end)
    result_df = all_seasons_df[['Game_Number', 'MATCHUP', 'SEASON', 'SEASON_ID', 'Centered_Avg']].copy()
    result_df['Centered_Avg_Stat'] = pd.to_numeric(result_df['Centered_Avg'], errors='coerce').round(2)
    result_df['STAT'] = value_col
    result_df = result_df.dropna(subset=['Centered_Avg_Stat'])

    data = result_df[['Game_Number','Centered_Avg_Stat','MATCHUP','SEASON','SEASON_ID','STAT']].to_dict(orient='records')

    output_dir = os.path.join(app.root_path, 'static')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'player_stats.json')  # same path your HTML reads

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {output_file} (rows: {len(result_df)}) for stat={value_col}")




def calculate_fantasy_points(row, scoring_rules):
    fpts = (row['FGM'] * scoring_rules['fgm'] +
            row['FGA'] * scoring_rules['fga'] +
            row['FTM'] * scoring_rules['ftm'] +
            row['FTA'] * scoring_rules['fta'] +
            row['FG3M'] * scoring_rules['threeptm'] +
            row['REB'] * scoring_rules['reb'] +
            row['AST'] * scoring_rules['ast'] +
            row['STL'] * scoring_rules['stl'] +
            row['BLK'] * scoring_rules['blk'] +
            row['TOV'] * scoring_rules['turno'] +
            row['PTS'] * scoring_rules['pts'])
    return fpts

@app.route('/compare_page', methods=['POST'])
def compare_page():
    start_time = time.time()
    try:
        # Attempt to convert form inputs to integers
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
        # Flash an error message to the user
        flash("Invalid input. Please ensure all stats are numbers.")
        
        # Retrieve league details to maintain context
        league_id = request.form.get('league_id')
        year = request.form.get('year')
        espn_s2 = request.form.get('espn_s2')
        swid = request.form.get('swid')
        scoring_type = request.form.get('scoring_type')
        
        # Construct the info string for redirection
        info_list = [league_id, year, espn_s2, swid]
        info_string = ','.join(filter(None, info_list))
        
        # Save the current form data to the session for reuse
        session['form_data'] = request.form.to_dict()
        
        # Redirect back to select_teams_page with league info
        return redirect(url_for('select_teams_page', info=info_string, scoring_type=scoring_type))
    
    # Retrieve the custom scoring values from the form
    league_scoring_rules = {
        'fgm': fgm,
        'fga': fga,
        'ftm': ftm,
        'fta': fta,
        'threeptm': threeptm,
        'reb': reb,
        'ast': ast,
        'stl': stl,
        'blk': blk,
        'turno': turno,
        'pts': pts,
    }
        
    my_team_name = request.form.get('myTeam')
    opponents_team_name = request.form.get('opponentsTeam')
    league_id = request.form.get('league_id')
    year = int(request.form.get('year'))
    espn_s2 = request.form.get('espn_s2')
    swid = request.form.get('swid')
    scoring_type = request.form.get('scoring_type')
    week_num = int(request.form.get('week_num'))
    
    print(f"Processing league with scoring type: {scoring_type}")
    
    try:
        if espn_s2 and swid:
            league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
        else:
            league = League(league_id=league_id, year=year)
            
        #print(f"League settings scoring type: {league.settings.scoring_type}")
        #(f"League settings: {vars(league.settings)}")
            
        # Create matchup data dictionary
        matchup_data_dict = get_matchup_dates(league)
        
        # Create week_data dictionary with the selected matchup period info
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
        return redirect(url_for('entry_page', error_message=error_message))
    except Exception as e:
        print(f"Error initializing league: {str(e)}")
        return redirect(url_for('entry_page', error_message=str(e)))

    team1_index = -1
    team2_index = -1

    count = 0
    for i in league.teams:
        if i.team_name == my_team_name:
            team1_index = count
        if i.team_name == opponents_team_name:
            team2_index = count
        count += 1

    if team1_index == -1:
        return redirect(url_for('entry_page', error_message="Team 1 not found. Please try again."))

    if team2_index == -1:
        return redirect(url_for('entry_page', error_message="Team 2 not found. Please try again."))
    
    player_data_column_names = ['player_name', 'min', 'fgm', 'fga', 'fg%', 'ftm', 'fta', 'ft%', 'threeptm', 'reb', 'ast', 'stl', 'blk', 'turno', 'pts', 'inj', 'fpts', 'games']

    if scoring_type == "H2H_POINTS":
        team1_player_data = cpd.get_team_player_data(league, team1_index, player_data_column_names, year, league_scoring_rules, week_data)
        team2_player_data = cpd.get_team_player_data(league, team2_index, player_data_column_names, year, league_scoring_rules, week_data)

        team_data_column_names = ['team_avg_fpts', 'team_expected_points', 'team_chance_of_winning', 'team_name', 'team_current_points']

        team1_data, team2_data = tsd.get_team_stats(league, team1_index, team1_player_data, team2_index, team2_player_data, team_data_column_names, league_scoring_rules, year, week_data)

        combined_df = cpd.get_compare_graph(league, team1_index, team1_player_data, team2_index, team2_player_data, year, week_data)
        combined_json = combined_df.to_json(orient='records')

        # Convert DataFrames to list of dictionaries
        team1_player_data = team1_player_data.to_dict(orient='records')
        team2_player_data = team2_player_data.to_dict(orient='records')
        team1_data = team1_data.to_dict(orient='records')
        team2_data = team2_data.to_dict(orient='records')

        return render_template('compare_page.html', 
                            data_team_players_1=team1_player_data, 
                            data_team_players_2=team2_player_data, 
                            data_team_stats_1=team1_data, 
                            data_team_stats_2=team2_data,
                            combined_json=combined_json,
                            scoring_type="H2H_POINTS",
                            week_data=week_data)
    
    elif scoring_type in ["H2H_CATEGORY", "H2H_MOST_CATEGORIES"]:  # Support both category types
        # Retrieve player data for both teams
        team1_player_data = cpd.get_team_player_data(
            league, team1_index, player_data_column_names, year, league_scoring_rules, week_data
        )
        team2_player_data = cpd.get_team_player_data(
            league, team2_index, player_data_column_names, year, league_scoring_rules, week_data
        )
        
        # Get team stats for categories
        team1_data, team2_data, team1_win_pcts, team2_win_pcts = tsd.get_team_stats_categories(
            league, team1_index, team1_player_data, team2_index, team2_player_data, 
            league_scoring_rules, year, week_data
        )
        
        print(week_data)

        # Generate comparison graphs for each category
        combined_dfs = cpd.get_compare_graphs_categories(
            league, team1_index, team1_player_data, team2_index, team2_player_data, year, week_data
        )

        #print(combined_dfs)
        
        combined_dicts = {cat: df.to_dict(orient='records') for cat, df in combined_dfs.items()}

        # Convert DataFrames to lists of dictionaries for rendering
        team1_player_data = team1_player_data.to_dict(orient='records')
        team2_player_data = team2_player_data.to_dict(orient='records')
        team1_data = team1_data.to_dict(orient='records')
        team2_data = team2_data.to_dict(orient='records')

        team1_win_pct_data = team1_win_pcts.to_dict(orient='records')
        team2_win_pct_data = team2_win_pcts.to_dict(orient='records')
        #print(combined_dicts)
        return render_template(
            'compare_page_cat.html', 
            data_team_players_1=team1_player_data, 
            data_team_players_2=team2_player_data, 
            data_team_stats_1=team1_data, 
            data_team_stats_2=team2_data,
            team1_win_pct_data=team1_win_pct_data,
            team2_win_pct_data=team2_win_pct_data,
            combined_jsons=combined_dicts,
            scoring_type=scoring_type,
            week_data=week_data
        )
    
    # Default case if scoring_type is not recognized
    print(f"Unrecognized scoring type: {scoring_type}")
    return redirect(url_for('entry_page', error_message=f"Unsupported scoring type: {scoring_type}"))

@app.route('/select_teams_page')
def select_teams_page():
    info_string = request.args.get('info', '')
    info_list = info_string.split(',') if info_string else []

    try:
        year = int(info_list[1])
    except:
        return redirect(url_for('entry_page', error_message="Invalid league entered. Please try again."))

    #check if user has input swid and espn_s2 so we can use the right league call (private vs public league)
    if len(info_list)==4:
        league_details = {
            'league_id': info_list[0],
            'year': year,
            'espn_s2': info_list[2],
            'swid': info_list[3]
        }
    else:
        league_details = {
            'league_id': info_list[0],
            'year': year,
            'espn_s2': None,
            'swid': None
        }
    
    #choose one based on if swid and espn_s2 are given
    if league_details['espn_s2'] == None:
        try:
            league = League(league_id=league_details['league_id'], year=league_details['year'])
        except (ESPNUnknownError, ESPNInvalidLeague):
            return redirect(url_for('entry_page', error_message="Invalid league entered. Please try again."))
        except ESPNAccessDenied:
            return redirect(url_for('entry_page', error_message="That is a private league which needs espn_s2 and SWID. Please try again."))
    else:
        try:
            league = League(league_id=league_details['league_id'], year=league_details['year'], espn_s2=league_details['espn_s2'], swid=league_details['swid'])
        except ESPNUnknownError:
            return redirect(url_for('entry_page', error_message="Invalid league entered. Please try again."))

    #if league.settings.scoring_type == "H2H_CATEGORY":
    #    return redirect(url_for('entry_page', error_message="League must be Points, not Categories"))

    # here i need to determine which week it currently is, and the start and end dates for each week, and period data

    # dates should be a data object, and dict should look like this:

    # week:matchupperiod, [scoringperiods], firstdate, lastdate

    # scoring preiods are calculate like this:
    # for i, date in enumerate(dates):
    #    scoring_period = i+(matchup_period-1)*7
    # 1 scoring period for each date in a matchup
    # week 1 is 1 day less because it starts on a tuesday instead of a monday
    # the starting date can be calculated by todays date - league.scoringPeriodId, because the leaguecats.scoringPeriodId is also the number of days since the season started.

    # matchup 17 - league.firstScoringPeriod is all star break, so it is 7 days longer. the break days also have their scoring period calculated the same way

    # use league.playoff_matchup_period_length determines the length of playoff matchups in # weeks (usually 2 weeks, 14 days), playoff matchups are the last matchup periods.

    #league.settings.matchup_periods is {'1': [1], '2': [2], '3': [3], '4': [4], '5': [5], '6': [6], '7': [7], '8': [8], '9': [9], '10': [10], '11': [11], '12': [12], '13': [13], '14': [14], '15': [15], '16': [16], '17': [17], '18': [18], '19': [19], '20': [20], '21': [21], '22': [22]}
    # for example

    #print(vars(league.settings))
    
    matchup_date_data = get_matchup_dates(league)


    teams_list = [team.team_name for team in league.teams]

    form_data = session.pop('form_data', {})

    return render_template('select_teams_page.html', 
                          info_list=teams_list, 
                          **league_details, 
                          form_data=form_data, 
                          scoring_type=league.settings.scoring_type,
                          matchup_data_dict=matchup_date_data,
                          current_matchup=league.currentMatchupPeriod)

@app.route('/process', methods=['POST'])
def process_information():
    league_id = request.form.get('league_id')
    year = request.form.get('year')
    espn_s2 = request.form.get('espn_s2')
    swid = request.form.get('swid')

    info_list = [league_id, year, espn_s2, swid]
    info_string = ','.join(filter(None, info_list))

    return redirect(url_for('select_teams_page', info=info_string))

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
