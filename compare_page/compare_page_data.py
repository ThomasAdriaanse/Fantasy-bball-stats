import pandas as pd
import db_utils
from datetime import datetime, timedelta


def get_team_player_data_schema():
    table_schema = """
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
    
    return table_schema


def get_team_player_data(league, team_num, table, columns):

    team = league.teams[team_num]
    
    # Initialize an empty list for each column
    team_data = {column: [] for column in columns}

    # Single loop to collect data
    for player in team.roster:
    # Check if player is already in the database
        if 'avg' in player.stats['2024_total'].keys():
            player_avg_stats = player.stats['2024_total']['avg']
            team_data['player_name'].append(player.name)
            team_data['min'].append(round(player_avg_stats['MIN'], 2))
            team_data['fgm'].append(round(player_avg_stats['FGM'], 2))
            team_data['fga'].append(round(player_avg_stats['FGA'], 2))
            team_data['ftm'].append(round(player_avg_stats['FTM'], 2))
            team_data['fta'].append(round(player_avg_stats['FTA'], 2))
            team_data['threeptm'].append(round(player_avg_stats['3PTM'], 2))
            team_data['reb'].append(round(player_avg_stats['REB'], 2))
            team_data['ast'].append(round(player_avg_stats['AST'], 2))
            team_data['stl'].append(round(player_avg_stats['STL'], 2))
            team_data['blk'].append(round(player_avg_stats['BLK'], 2))
            team_data['turno'].append(round(player_avg_stats['TO'], 2))
            team_data['pts'].append(round(player_avg_stats['PTS'], 2))
            team_data['inj'].append(player.injuryStatus)

            fpts = round(player_avg_stats['FGM']*2 - player_avg_stats['FGA'] + player_avg_stats['FTM'] - 
            player_avg_stats['FTA'] + player_avg_stats['3PTM'] + player_avg_stats['REB'] + 
            2*player_avg_stats['AST'] + 4*player_avg_stats['STL'] + 4*player_avg_stats['BLK'] - 
            2*player_avg_stats['TO'] + player_avg_stats['PTS'], 2)
            
            team_data['fpts'].append(fpts)


            #get the players schedule this week
            schedule = player.schedule
            list_schedule = list(schedule.values())
            list_schedule.sort(key=lambda x: x['date'], reverse=False)

            start_of_week, end_of_week = db_utils.range_of_current_week()

            #games_this_week = [game for game in list_schedule if start_of_week <= game['date'] <= end_of_week]

            today = datetime.today()
            games_left_this_week = [game for game in list_schedule if today <= game['date'] <= end_of_week]
            
            #add number of games left to the database
            team_data['games'].append(len(games_left_this_week))

    df = pd.DataFrame(team_data)
    return df

