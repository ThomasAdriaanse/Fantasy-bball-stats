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
        games_left_this_week = [game for game in list_schedule if today <= game['date'] <= end_of_week+timedelta(hours=9)]
        
        #add number of games left to the database
        team_data['games'].append(len(games_left_this_week))

    df = pd.DataFrame(team_data)
    return df

def get_compare_graph(league, team1_index, team1_player_data, team2_index, team2_player_data):
    
    team1 = league.teams[team1_index]
    team2 = league.teams[team2_index]

    today = datetime.today().date()
    start_of_week, end_of_week =db_utils.range_of_current_week() 

    dates = pd.date_range(start=start_of_week, end=end_of_week).date

    # Need this dict for the function to tell which scores correspond to which dates in the predicted_values arrays
    dates_dict = {date: i for i, date in enumerate(dates)}

    # Initialize dictionaries to store predicted and actual FPTS for both teams
    predicted_values_team1 = []
    predicted_values_from_present_team1 = []
    predicted_values_team2 = []
    predicted_values_from_present_team2 = []


    for date in dates:
        predicted_values_team1.append(0)
        predicted_values_from_present_team1.append(0)
        predicted_values_team2.append(0)
        predicted_values_from_present_team2.append(0)

    # Helper function to calculate FPTS for a team's roster
    def calculate_fpts_for_team(team, team_player_data, predicted_values, predicted_values_from_present, dates_dict):

        for player in team.roster:
            player_name = player.name
            player_row = team_player_data[team_player_data['player_name'] == player_name]

            if player_row.empty:
                avg_fpts = 0  # If no matching player is found in player_data, set avg_fpts to 0
            else:
                avg_fpts = player_row['fpts'].values[0]

            # For testing
            #avg_fpts = random.randint(20, 60)

            if not isinstance(avg_fpts, str) and player_row['inj'].values[0] != 'OUT':
                # Get player's schedule
                list_schedule = list(player.schedule.values())
                list_schedule.sort(key=lambda x: x['date'], reverse=False)

                # Add FPTS to dates when appropriate, convert to pst from utc 0
                for game in list_schedule:
                    game_date = (game['date']-timedelta(hours=9)).date()

                    if game_date in dates_dict:
                        predicted_values[dates_dict[game_date]] += avg_fpts

                        if game_date >= today:
                            predicted_values_from_present[dates_dict[game_date]] += avg_fpts

    #print(predicted_values_from_present_team1)

    # Calculate FPTS for both teams
    calculate_fpts_for_team(team1, team1_player_data, predicted_values_team1, predicted_values_from_present_team1, dates_dict)
    calculate_fpts_for_team(team2, team2_player_data, predicted_values_team2, predicted_values_from_present_team2, dates_dict)

    boxscore_number_team1, home_or_away_team1 = get_team_boxscore_number(league, team1)
    boxscore_number_team2, home_or_away_team2 = get_team_boxscore_number(league, team2)

    box_scores = league.box_scores(matchup_total=False)

    #new
    team1_box_score_list = []
    team2_box_score_list = []

    #creating lists of the weekly box scores for each team
    for i, date in enumerate(dates):

        if home_or_away_team1 == "home":
            team1_box_score_list.append(league.box_scores(scoring_period=i, matchup_total=False)[boxscore_number_team1].home_score)
        if home_or_away_team1 == "away":
            team1_box_score_list.append(league.box_scores(scoring_period=i, matchup_total=False)[boxscore_number_team1].away_score)

        if home_or_away_team2 == "home":
            team2_box_score_list.append(league.box_scores(scoring_period=i, matchup_total=False)[boxscore_number_team2].home_score)
        if home_or_away_team2 == "away":
            team2_box_score_list.append(league.box_scores(scoring_period=i, matchup_total=False)[boxscore_number_team2].away_score)

    print(team1_box_score_list)
    print(team2_box_score_list)

    # Instead of 0s ESPN uses the moost recent score, so i have to replace scores that maatch the most recent score with 0
    # I can append the most recent score later
    #for i, v in enumerate(team1_box_score_list):
    #    if team1_box_score_list[i]==team1_box_score_list[-1]:
    #        team1_box_score_list[i]-=team1_box_score_list[-1]

    #    if team2_box_score_list[i]==team2_box_score_list[-1]:
    #        team2_box_score_list[i]-=team1_box_score_list[-1]

    #old in tact
    for index, date in enumerate(dates):

        box_scores = league.box_scores(scoring_period=index, matchup_total=False)
        
        if date < today:
            if home_or_away_team1 == "home":
                predicted_values_from_present_team1[index] = team1_box_score_list[index]
            elif home_or_away_team1 == "away":
                predicted_values_from_present_team1[index] = team1_box_score_list[index]

            if home_or_away_team2 == "home":
                predicted_values_from_present_team2[index] = team2_box_score_list[index]
            elif home_or_away_team2 == "away":
                predicted_values_from_present_team2[index] = team2_box_score_list[index]


        if index > 0:
            predicted_values_team1[index] += predicted_values_team1[index-1]
            predicted_values_team2[index] += predicted_values_team2[index-1]

            predicted_values_from_present_team1[index] += predicted_values_from_present_team1[index-1]
            predicted_values_from_present_team2[index] += predicted_values_from_present_team2[index-1]

    print(predicted_values_from_present_team1)
    print(predicted_values_from_present_team2)


    # Convert the predicted values into separate DataFrames for each team
    team1_df = pd.DataFrame({
        'date': dates,
        'predicted_fpts': predicted_values_team1,
        'predicted_fpts_from_present': predicted_values_from_present_team1,
        'team': 'Team 1'
    })

    team2_df = pd.DataFrame({
        'date': dates,
        'predicted_fpts': predicted_values_team2,
        'predicted_fpts_from_present': predicted_values_from_present_team2,
        'team': 'Team 2'
    })

    # Combine both DataFrames
    combined_df = pd.concat([team1_df, team2_df], ignore_index=True)

    return combined_df

def get_current_score(league, team):

    for boxscore in league.box_scores():
        if team == boxscore.home_team:
            return boxscore.home_score 

        elif team == boxscore.away_team:  
            return boxscore.away_score 

    return "error"

def get_team_boxscore_number(league, team):

    for index, boxscore in enumerate(league.box_scores(matchup_total=False)):
        #print(boxscore)
        #print(index)
        if team == boxscore.home_team:
         #   print("in")
            return index, "home"

        elif team == boxscore.away_team:  
         #   print("in")
            return index, "away"

    return "error"