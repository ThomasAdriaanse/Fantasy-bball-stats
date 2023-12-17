from flask import Flask, render_template
import psycopg2
import os
from dotenv import load_dotenv
from psycopg2 import extras
import db_utils
import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd
from scipy.stats import norm
from espn_api.basketball import League
import flask
import pandas as pd



load_dotenv() 
app = Flask(__name__)


@app.route('/')
def entry_page():
    # Render the entry page template on start
    return render_template('entry.html')


@app.route('/compare_page', methods=['POST'])
def compare_page():

    my_team_name = flask.request.form.get('myTeam')
    opponents_team_name = flask.request.form.get('opponentsTeam')

    league_id = flask.request.form.get('league_id')
    year = int(flask.request.form.get('year'))
    espn_s2 = flask.request.form.get('espn_s2')
    swid = flask.request.form.get('swid')

    league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)

    print(my_team_name, opponents_team_name)

    team1_index = -1
    team2_index = -1

    count = 0
    for i in league.teams:
        if i.team_name == my_team_name:
            team1_index = count

        if i.team_name == opponents_team_name:
            team2_index = count

        count+=1
    
    if team1_index == -1:
        print("error: team 1 not found")
        return
    
    if team2_index == -1:
        print("error: team 2 not found")
        return
    
    print(team1_index, team2_index)


    compare_table_name_1 = 'compare_team_players_1'
    compare_table_name_2 = 'compare_team_players_2'
    team_stats_table_name_1 = 'compare_team_stats_1'
    team_stats_table_name_2 = 'compare_team_stats_2'

    conn = psycopg2.connect(**db_utils.get_connection_parameters())
    cur = conn.cursor()

    db_utils.drop_table(cur, compare_table_name_1)
    db_utils.drop_table(cur, compare_table_name_2)
    db_utils.drop_table(cur, team_stats_table_name_1)
    db_utils.drop_table(cur, team_stats_table_name_2)

    db_utils.create_table(cur, compare_table_name_1, cpd.get_team_player_data_schema())
    db_utils.create_table(cur, compare_table_name_2, cpd.get_team_player_data_schema())
    db_utils.create_table(cur, team_stats_table_name_1, tsd.get_team_stats_data_schema())
    db_utils.create_table(cur, team_stats_table_name_2, tsd.get_team_stats_data_schema())

    conn.commit()
    cur.close()
    conn.close()

    player_data_column_names = ['player_name', 'min', 'fgm', 'fga', 'ftm', 'fta', 'threeptm', 'reb', 'ast', 'stl', 'blk', 'turno', 'pts', 'inj','fpts', 'games']

    team1_player_data = cpd.get_team_player_data(league, team1_index, compare_table_name_1, player_data_column_names)
    db_utils.insert_data_to_db(team1_player_data, compare_table_name_1, player_data_column_names)

    team2_player_data = cpd.get_team_player_data(league, team2_index, compare_table_name_2, player_data_column_names)
    db_utils.insert_data_to_db(team2_player_data, compare_table_name_2, player_data_column_names)

    team_data_column_names = ['team_avg_fpts', 'team_expected_points', 'team_chance_of_winning', 'team_name', 'team_current_points']

    team1_data = tsd.get_team_stats(league, team1_index, team1_player_data, team2_index, team2_player_data, team_data_column_names)
    db_utils.insert_data_to_db(team1_data, team_stats_table_name_1, team_data_column_names)

    team2_data = tsd.get_team_stats(league, team2_index, team2_player_data, team1_index, team1_player_data, team_data_column_names)
    db_utils.insert_data_to_db(team2_data, team_stats_table_name_2, team_data_column_names)


    conn = db_utils.get_db_connection()
    cur = conn.cursor(cursor_factory=extras.DictCursor)

    cur.execute('SELECT * FROM compare_team_players_1;')
    data_team_players_1 = cur.fetchall()

    cur.execute('SELECT * FROM compare_team_players_2;')
    data_team_players_2 = cur.fetchall()

    cur.execute('SELECT * FROM compare_team_stats_1;')
    data_team_stats_1 = cur.fetchall()

    cur.execute('SELECT * FROM compare_team_stats_2;')
    data_team_stats_2 = cur.fetchall()

    cur.close()
    conn.close()

    # Pass both datasets to the template
    return render_template('compare_page.html', data_team_players_1=data_team_players_1, data_team_players_2=data_team_players_2, data_team_stats_1=data_team_stats_1, data_team_stats_2=data_team_stats_2)


@app.route('/select_teams_page')
def select_teams_page():
    info_string = flask.request.args.get('info', '')
    info_list = info_string.split(',') if info_string else []

    print(info_list)

    league_details = {
        'league_id': info_list[0],
        'year': int(info_list[1]),
        'espn_s2': info_list[2],
        'swid': info_list[3]
    }
    
    league = League(league_id=league_details['league_id'], year=league_details['year'], espn_s2=league_details['espn_s2'], swid=league_details['swid'])

    teams_list = [team.team_name for team in league.teams]

    # Include the league details in the context sent to the template
    return render_template('select_teams_page.html', info_list=teams_list, **league_details)


@app.route('/process', methods=['POST'])
def process_information():
    league_id = flask.request.form.get('league_id')
    year = flask.request.form.get('year')
    espn_s2 = flask.request.form.get('espn_s2')
    swid = flask.request.form.get('swid')


    # Combine the information into a list
    info_list = [league_id, year, espn_s2, swid]

    info_string = ','.join(filter(None, info_list))  # This will join the list into a string, separating items by commas

    return flask.redirect(flask.url_for('select_teams_page', info=info_string))



if __name__ == '__main__':
    
    '''compare_table_name_1 = 'compare_team_players_1'
    compare_table_name_2 = 'compare_team_players_2'
    team_stats_table_name_1 = 'compare_team_stats_1'
    team_stats_table_name_2 = 'compare_team_stats_2'

    conn = psycopg2.connect(**db_utils.get_connection_parameters())
    cur = conn.cursor()

    db_utils.drop_table(cur, compare_table_name_1)
    db_utils.drop_table(cur, compare_table_name_2)
    db_utils.drop_table(cur, team_stats_table_name_1)
    db_utils.drop_table(cur, team_stats_table_name_2)

    db_utils.create_table(cur, compare_table_name_1, cpd.get_team_player_data_schema())
    db_utils.create_table(cur, compare_table_name_2, cpd.get_team_player_data_schema())
    db_utils.create_table(cur, team_stats_table_name_1, tsd.get_team_stats_data_schema())
    db_utils.create_table(cur, team_stats_table_name_2, tsd.get_team_stats_data_schema())

    conn.commit()
    cur.close()
    conn.close()'''

    print(pd. __version__)

    # Start the Flask application
    #app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

    