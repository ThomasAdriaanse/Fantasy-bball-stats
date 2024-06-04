from flask import Flask, render_template, redirect, url_for, request
import psycopg2
import os
from dotenv import load_dotenv
from psycopg2 import extras
import db_utils
import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd
from scipy.stats import norm
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError
import pandas as pd

load_dotenv()
app = Flask(__name__)

@app.route('/')
def entry_page():
    error_message = request.args.get('error_message', '')
    return render_template('entry.html', error_message=error_message)


@app.route('/compare_page', methods=['POST'])
def compare_page():
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

    compare_table_name_1 = 'compare_team_players_1'
    compare_table_name_2 = 'compare_team_players_2'
    team_stats_table_name_1 = 'compare_team_stats_1'
    team_stats_table_name_2 = 'compare_team_stats_2'

    player_data_column_names = ['player_name', 'min', 'fgm', 'fga', 'ftm', 'fta', 'threeptm', 'reb', 'ast', 'stl', 'blk', 'turno', 'pts', 'inj', 'fpts', 'games']

    team1_player_data = cpd.get_team_player_data(league, team1_index, compare_table_name_1, player_data_column_names)
    team2_player_data = cpd.get_team_player_data(league, team2_index, compare_table_name_2, player_data_column_names)

    team_data_column_names = ['team_avg_fpts', 'team_expected_points', 'team_chance_of_winning', 'team_name', 'team_current_points']

    team1_data = tsd.get_team_stats(league, team1_index, team1_player_data, team2_index, team2_player_data, team_data_column_names)
    team2_data = tsd.get_team_stats(league, team2_index, team2_player_data, team1_index, team1_player_data, team_data_column_names)

    return render_template('compare_page.html', data_team_players_1=team1_player_data, data_team_players_2=team2_player_data, data_team_stats_1=team1_data, data_team_stats_2=team2_data)

@app.route('/select_teams_page')
def select_teams_page():
    info_string = request.args.get('info', '')
    info_list = info_string.split(',') if info_string else []

    league_details = {
        'league_id': info_list[0],
        'year': int(info_list[1]),
        'espn_s2': info_list[2],
        'swid': info_list[3]
    }

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
    print(pd.__version__)

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
