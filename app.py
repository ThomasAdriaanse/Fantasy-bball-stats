from flask import Flask, render_template, redirect, url_for, request
import os
from dotenv import load_dotenv
from psycopg2 import extras
import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd
from scipy.stats import norm
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import json
import time

load_dotenv()
app = Flask(__name__)

@app.route('/')
def entry_page():
    error_message = request.args.get('error_message', '')
    return render_template('entry.html', error_message=error_message)

@app.route('/player_stats', methods=['GET', 'POST'])
def player_stats():
    num_games = 20  # Default to 20 games
    
    # Check if player_name is passed via query parameters (GET request)
    if request.method == 'GET':
        selected_player = request.args.get('player_name', 'Bol Bol')  # Default to 'bol bol' if no player_name is given
    elif request.method == 'POST':
        selected_player = request.form.get('player_name', 'Bol Bol')  # Default player if no input
        num_games = int(request.form.get('num_games', num_games))  # Update based on user input
    
    # Take this away when pushing
    #generate_json_file(selected_player, num_games)  # Generate JSON file with specified player and number of games

    # Get the list of all active players for the dropdown
    active_players = players.get_active_players()
    active_players.sort(key=lambda x: x['full_name'])

    return render_template('player_stats.html', players=active_players, selected_player=selected_player, num_games=num_games)


# Route to serve the JSON file for player stats
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('static', filename)

# Player stats generation function
def generate_json_file(player_name, num_games):

    player_info = players.find_players_by_full_name(player_name)
    
    if not player_info:
        print(f"Player {player_name} not found.")
        return
    
    player_id = player_info[0]['id']
    print(f"Fetching data for: {player_name} (ID: {player_id})")

    all_seasons_df = pd.DataFrame()

    year = 2023
    while year >= 1999:
        seasons_checked = 0
        while seasons_checked < 3:
            season = str(year)
            print(f"Fetching data for season {season}...")

            game_log = playergamelog.PlayerGameLog(player_id, season=season, season_type_all_star="Regular Season").get_dict()

            if game_log['resultSets'][0]['rowSet']:
                games = game_log['resultSets'][0]['rowSet']
                df = pd.DataFrame(games, columns=game_log['resultSets'][0]['headers'])

                all_seasons_df = pd.concat([all_seasons_df, df], ignore_index=True)
                break
            else:
                print(f"No data found for season {season}. Checking next year...")
                seasons_checked += 1
                year -= 1
        
        if seasons_checked == 3:
            print(f"No games found for three consecutive seasons starting from {year + 3}. Stopping.")
            break
        
        year -= 1
    
    if all_seasons_df.empty:
        print("No data available for the specified player.")
        return

    league_scoring_rules = {
        'fgm': 2,
        'fga': -1,
        'ftm': 1,
        'fta': -1,
        'threeptm': 1,
        'reb': 1,
        'ast': 2,
        'stl': 4,
        'blk': 4,
        'turno': -2,
        'pts': 1
    }

    all_seasons_df['FPTS'] = all_seasons_df.apply(lambda row: calculate_fantasy_points(row, league_scoring_rules), axis=1)

    all_seasons_df['Game_Date'] = pd.to_datetime(all_seasons_df['GAME_DATE'], format='%b %d, %Y')
    all_seasons_df = all_seasons_df.sort_values('Game_Date').reset_index(drop=True)

    all_seasons_df['Game_Number'] = all_seasons_df.index + 1

    def centered_average(index, x):
        start = max(0, index - x)
        end = min(len(all_seasons_df), index + x + 1)
        return all_seasons_df['FPTS'].iloc[start:end].mean()

    all_seasons_df['Centered_Avg_FPTS'] = all_seasons_df.index.map(lambda i: centered_average(i, num_games))

    result_df = all_seasons_df[['Game_Number', 'Centered_Avg_FPTS', 'MATCHUP']].replace({float('nan'): None}).dropna()

    data = result_df.to_dict(orient='records')

    output_dir = os.path.join(app.root_path, 'static')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'player_fpts.json')
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Data saved to {output_file}")

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

    # Retrieve the custom scoring values from the form
    league_scoring_rules = {
        'fgm': int(request.form.get('fgm', 2)),
        'fga': int(request.form.get('fga', -1)),
        'ftm': int(request.form.get('ftm', 1)),
        'fta': int(request.form.get('fta', -1)),
        'threeptm': int(request.form.get('threeptm', 1)),
        'reb': int(request.form.get('reb', 1)),
        'ast': int(request.form.get('ast', 2)),
        'stl': int(request.form.get('stl', 4)),
        'blk': int(request.form.get('blk', 4)),
        'turno': int(request.form.get('turno', -2)),
        'pts': int(request.form.get('pts', 1)),
    }
        
    my_team_name = request.form.get('myTeam')
    opponents_team_name = request.form.get('opponentsTeam')
    league_id = request.form.get('league_id')
    year = int(request.form.get('year'))
    espn_s2 = request.form.get('espn_s2')
    swid = request.form.get('swid')

    try:
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
    except ESPNUnknownError:
        return redirect(url_for('entry_page', error_message="Invalid league entered. Please try again."))

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

    current_matchup_period = league.currentMatchupPeriod

    player_data_column_names = ['player_name', 'min', 'fgm', 'fga', 'ftm', 'fta', 'threeptm', 'reb', 'ast', 'stl', 'blk', 'turno', 'pts', 'inj', 'fpts', 'games']

    init_data_time = time.time()
    print(f"Time to initialize data: {init_data_time - start_time:.2f} seconds")

    team1_player_data = cpd.get_team_player_data(league, team1_index, player_data_column_names, league_scoring_rules, year)
    team2_player_data = cpd.get_team_player_data(league, team2_index, player_data_column_names, league_scoring_rules, year)

    get_player_data_time = time.time()
    print(f"Time to get player data: {get_player_data_time - init_data_time:.2f} seconds")

    team_data_column_names = ['team_avg_fpts', 'team_expected_points', 'team_chance_of_winning', 'team_name', 'team_current_points']

    team1_data, team2_data = tsd.get_team_stats(league, team1_index, team1_player_data, team2_index, team2_player_data, team_data_column_names, league_scoring_rules, year)
    #team2_data = tsd.get_team_stats(league, team2_index, team2_player_data, team1_index, team1_player_data, team_data_column_names, league_scoring_rules, year)

    get_team_stats_time = time.time()
    print(f"Time to get team stats: {get_team_stats_time - get_player_data_time:.2f} seconds")

    combined_df = cpd.get_compare_graph(league, team1_index, team1_player_data, team2_index, team2_player_data, year)
    combined_json = combined_df.to_json(orient='records')  # Convert the DataFrame to JSON
    print(combined_df)

    # Convert DataFrames to list of dictionaries
    team1_player_data = team1_player_data.to_dict(orient='records')
    team2_player_data = team2_player_data.to_dict(orient='records')
    team1_data = team1_data.to_dict(orient='records')
    team2_data = team2_data.to_dict(orient='records')

    end_time = time.time()
    print(f"Total time: {end_time - get_team_stats_time:.2f} seconds")

    return render_template('compare_page.html', 
                            data_team_players_1=team1_player_data, 
                            data_team_players_2=team2_player_data, 
                            data_team_stats_1=team1_data, 
                            data_team_stats_2=team2_data,
                            combined_json=combined_json)


@app.route('/select_teams_page')
def select_teams_page():
    info_string = request.args.get('info', '')
    info_list = info_string.split(',') if info_string else []

    #check if user has input swid and espn_s2 so we can use the right league call (private vs public league)
    if len(info_list)==4:
        league_details = {
            'league_id': info_list[0],
            'year': int(info_list[1]),
            'espn_s2': info_list[2],
            'swid': info_list[3]
        }
    else:
        league_details = {
            'league_id': info_list[0],
            'year': int(info_list[1]),
            'espn_s2': 1,
            'swid': None
        }
    
    #choose one based on if swid and espn_s2 are given
    if league_details['espn_s2'] == 1:
        try:
            league = League(league_id=league_details['league_id'], year=league_details['year'])
        except ESPNUnknownError:
            return redirect(url_for('entry_page', error_message="Invalid league entered. Please try again."))
    else:
        try:
            league = League(league_id=league_details['league_id'], year=league_details['year'], espn_s2=league_details['espn_s2'], swid=league_details['swid'])
        except ESPNUnknownError:
            return redirect(url_for('entry_page', error_message="Invalid league entered. Please try again."))

    teams_list = [team.team_name for team in league.teams]

    return render_template('select_teams_page.html', info_list=teams_list, **league_details)


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
