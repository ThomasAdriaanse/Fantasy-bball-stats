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


def get_team_player_data(league, team_num, columns, league_scoring_rules, year):

    team = league.teams[team_num]
    
    # Initialize an empty list for each column
    team_data = {column: [] for column in columns}

    year_string = str(year) + "_total"

    # Single loop to collect data
    for player in team.roster:
    # Check if player is already in the database
        if year_string in player.stats and 'avg' in player.stats[year_string].keys():
            player_avg_stats = player.stats[year_string]['avg']
            team_data['player_name'].append(player.name)
            team_data['min'].append(round(player_avg_stats['MIN'], 2))
            team_data['fgm'].append(round(player_avg_stats['FGM'], 2))
            team_data['fga'].append(round(player_avg_stats['FGA'], 2))
            team_data['ftm'].append(round(player_avg_stats['FTM'], 2))
            team_data['fta'].append(round(player_avg_stats['FTA'], 2))
            team_data['threeptm'].append(round(player_avg_stats['3PM'], 2))
            team_data['reb'].append(round(player_avg_stats['REB'], 2))
            team_data['ast'].append(round(player_avg_stats['AST'], 2))
            team_data['stl'].append(round(player_avg_stats['STL'], 2))
            team_data['blk'].append(round(player_avg_stats['BLK'], 2))
            team_data['turno'].append(round(player_avg_stats['TO'], 2))
            team_data['pts'].append(round(player_avg_stats['PTS'], 2))
            team_data['inj'].append(player.injuryStatus)

            fpts = round(
            player_avg_stats['FGM']*league_scoring_rules['fgm'] +
            player_avg_stats['FGA']*league_scoring_rules['fga'] + 
            player_avg_stats['FTM']*league_scoring_rules['ftm'] +
            player_avg_stats['FTA']*league_scoring_rules['fta'] + 
            player_avg_stats['3PM']*league_scoring_rules['threeptm'] + 
            player_avg_stats['REB']*league_scoring_rules['reb'] + 
            player_avg_stats['AST']*league_scoring_rules['ast'] + 
            player_avg_stats['STL']*league_scoring_rules['stl'] + 
            player_avg_stats['BLK']*league_scoring_rules['blk'] + 
            player_avg_stats['TO']*league_scoring_rules['turno'] + 
            player_avg_stats['PTS']*league_scoring_rules['pts']
            , 2)
            
            team_data['fpts'].append(fpts)

        else:
            team_data['player_name'].append(player.name)
            team_data['min'].append('N/A')
            team_data['fgm'].append('N/A')
            team_data['fga'].append('N/A')
            team_data['ftm'].append('N/A')
            team_data['fta'].append('N/A')
            team_data['threeptm'].append('N/A')
            team_data['reb'].append('N/A')
            team_data['ast'].append('N/A')
            team_data['stl'].append('N/A')
            team_data['blk'].append('N/A')
            team_data['turno'].append('N/A')
            team_data['pts'].append('N/A')
            team_data['inj'].append(player.injuryStatus)
            team_data['fpts'].append('N/A')
        
        #get the players schedule this week
        schedule = player.schedule
        list_schedule = list(schedule.values())
        list_schedule.sort(key=lambda x: x['date'], reverse=False)

        start_of_week, end_of_week = db_utils.range_of_current_week()

        today = datetime.today()
        games_left_this_week = [game for game in list_schedule if today <= game['date'] <= end_of_week]
        
        #add number of games left to the database
        team_data['games'].append(len(games_left_this_week))

    df = pd.DataFrame(team_data)
    return df


def get_compare_graph(league, team1_index, team1_player_data, team2_index, team2_player_data, team_data_column_names):
    team1 = league.teams[team1_index]
    team2 = league.teams[team2_index]

    today = datetime.today().date()
    start_of_week, end_of_week = db_utils.range_of_current_week()  # Assuming this function returns datetime.date objects

    # Initialize DataFrame for storing day-wise FPTS contributions
    dates = pd.date_range(start=start_of_week, end=end_of_week).date
    team1_daily_fpts = pd.DataFrame({'date': dates, 'predicted_fpts': [0] * len(dates), 'actual_fpts': [0] * len(dates)})
    team2_daily_fpts = team1_daily_fpts.copy()

    def calculate_daily_fpts(team, daily_fpts_df, player_data):
        for player in team.roster:
            player_name = player.name

            # Ensure that 'player_data' is a DataFrame and contains the player_name column
            # Filter to get the player's average FPTS from the DataFrame
            player_row = player_data[player_data['player_name'] == player_name]

            if player_row.empty:
                avg_fpts = 0  # If no matching player is found in player_data, set avg_fpts to 0
            else:
                avg_fpts = player_row['fpts'].values[0]

            # Get the player's schedule and sort by date
            player_schedule = player.schedule
            games_left = [game['date'] for game in player_schedule.values() if start_of_week <= game['date'] <= end_of_week]

            # Add FPTS contribution to each game day
            for game_day in games_left:
                daily_fpts_df.loc[daily_fpts_df['date'] == game_day, 'predicted_fpts'] += avg_fpts

            # If today is on or after a game day, contribute to 'actual_fpts' to reflect today's actual progress
            for game_day in games_left:
                if today >= game_day:
                    daily_fpts_df.loc[daily_fpts_df['date'] == game_day, 'actual_fpts'] += avg_fpts


    # Calculate contributions for Team 1
    calculate_daily_fpts(team1, team1_daily_fpts, team1_player_data)

    # Calculate contributions for Team 2
    calculate_daily_fpts(team2, team2_daily_fpts, team2_player_data)

    # Combine both teams' data into a single DataFrame for easy graphing
    team1_daily_fpts['team'] = 'Team 1'
    team2_daily_fpts['team'] = 'Team 2'
    combined_df = pd.concat([team1_daily_fpts, team2_daily_fpts], ignore_index=True)

    return combined_df

