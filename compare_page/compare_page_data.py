import pandas as pd
import db_utils
from datetime import datetime, timedelta
import time

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


def get_team_player_data(league, team_num, columns, year, league_scoring_rules, week_data=None):

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
            team_data['fg%'].append(round((player_avg_stats['FGM']*100)/player_avg_stats['FGA'], 2))
            team_data['ftm'].append(round(player_avg_stats['FTM'], 2))
            team_data['fta'].append(round(player_avg_stats['FTA'], 2))
            team_data['ft%'].append(round((player_avg_stats['FTM']*100)/player_avg_stats['FTA'], 2))
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
            team_data['fg%'].append('N/A')
            team_data['ftm'].append('N/A')
            team_data['fta'].append('N/A')
            team_data['ft%'].append('N/A')
            team_data['threeptm'].append('N/A')
            team_data['reb'].append('N/A')
            team_data['ast'].append('N/A')
            team_data['stl'].append('N/A')
            team_data['blk'].append('N/A')
            team_data['turno'].append('N/A')
            team_data['pts'].append('N/A')
            team_data['inj'].append(player.injuryStatus)
            team_data['fpts'].append('N/A')
        
        # Get the players schedule for the week
        schedule = player.schedule
        list_schedule = list(schedule.values())
        list_schedule.sort(key=lambda x: x['date'], reverse=False)

        # Use selected week data if provided, otherwise use current week
        if week_data and 'matchup_data' in week_data:
            # Use the selected week's date range from matchup_data
            start_date = datetime.strptime(week_data['matchup_data']['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(week_data['matchup_data']['end_date'], '%Y-%m-%d').date()
            
            # Filter games for the selected week
            games_in_week = [game for game in list_schedule if 
                            start_date <= (game['date']-timedelta(hours=5)).date() <= end_date]
        else:
            print("error: not using proper week data")
            # Fall back to current week if no week_data provided
            today_minus_8 = (datetime.today()-timedelta(hours=8)).date()
            start_of_week, end_of_week = db_utils.range_of_current_week(today_minus_8)
            
            # Filter games for current week
            games_in_week = [game for game in list_schedule if 
                            today_minus_8 <= (game['date']-timedelta(hours=5)).date() <= end_of_week]
        
        # Add number of games to the database
        team_data['games'].append(len(games_in_week))

    df = pd.DataFrame(team_data)
    return df

def get_compare_graph(league, team1_index, team1_player_data, team2_index, team2_player_data, year, week_data=None):
    
    start_time = time.time()
    
    team1 = league.teams[team1_index]
    team2 = league.teams[team2_index]

    # Use selected week data if provided, otherwise use current week
    if week_data and 'matchup_data' in week_data:
        # Use the selected week's date range and matchup period
        start_date = datetime.strptime(week_data['matchup_data']['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(week_data['matchup_data']['end_date'], '%Y-%m-%d').date()
        selected_matchup_period = week_data['selected_week']
    else:
        # Fall back to current week if no week_data provided
        today = datetime.today()
        today_minus_8 = (today-timedelta(hours=8)).date()
        start_date, end_date = db_utils.range_of_current_week(today_minus_8)
        selected_matchup_period = league.currentMatchupPeriod

    # Create date range for the selected week
    dates = pd.date_range(start=start_date, end=end_date+timedelta(hours=24)).date
    
    #print(dates)

    # Step 3: Initialize Dictionaries for Predicted Values
    dates_dict = {date: i for i, date in enumerate(dates)}
    predicted_values_team1 = [0] * len(dates)
    predicted_values_from_present_team1 = [0] * len(dates)
    predicted_values_team2 = [0] * len(dates)
    predicted_values_from_present_team2 = [0] * len(dates)
    
    # Helper Function to Calculate FPTS
    def calculate_fpts_for_team(team, team_player_data, predicted_values, predicted_values_from_present, dates_dict):
        for player in team.roster:
            player_name = player.name
            player_row = team_player_data[team_player_data['player_name'] == player_name]

            if player_row.empty:
                avg_fpts = 0
            else:
                avg_fpts = player_row['fpts'].values[0]

            if not isinstance(avg_fpts, str) and player_row['inj'].values[0] != 'OUT':
                list_schedule = list(player.schedule.values())
                list_schedule.sort(key=lambda x: x['date'], reverse=False)

                for game in list_schedule:# timezone adjustment
                    game_date = (game['date'] - timedelta(hours=5)).date()
                    if game_date in dates_dict:
                        predicted_values[dates_dict[game_date]] += avg_fpts
                        today_minus_8 = (datetime.today()-timedelta(hours=8)).date()
                        if game_date >= today_minus_8:
                            predicted_values_from_present[dates_dict[game_date]] += avg_fpts

    calculate_fpts_for_team(team1, team1_player_data, predicted_values_team1, predicted_values_from_present_team1, dates_dict)
    calculate_fpts_for_team(team2, team2_player_data, predicted_values_team2, predicted_values_from_present_team2, dates_dict)

    boxscore_number_team1, home_or_away_team1 = get_team_boxscore_number(league, team1, selected_matchup_period)
    boxscore_number_team2, home_or_away_team2 = get_team_boxscore_number(league, team2, selected_matchup_period)

    # Get Weekly Box Scores for Both Teams
    team1_box_score_list = []
    team2_box_score_list = []

    # Get scoring periods from week_data if available
    if week_data and 'matchup_data' in week_data:
        scoring_periods = week_data['matchup_data']['scoring_periods']
    else:
        matchup_periods = db_utils.get_matchup_periods(league, selected_matchup_period)
        scoring_periods = []
        for matchup_period in matchup_periods:
            for i, date in enumerate(dates):
                scoring_periods.append(i + (matchup_period - 1) * 7)

    # Initialize box score lists with zeros
    team1_box_score_list = [0] * len(dates)
    team2_box_score_list = [0] * len(dates)

    # Get historical data for dates that have already passed
    today_minus_8 = (datetime.today()-timedelta(hours=8)).date()
    for i, date in enumerate(dates):
        if date < today_minus_8:
            try:
                # Use scoring period from the matchup_data if available
                scoring_period = scoring_periods[i] if i < len(scoring_periods) else i + (selected_matchup_period - 1) * 7

                box_scores = league.box_scores(matchup_period=selected_matchup_period, 
                                              scoring_period=scoring_period, 
                                              matchup_total=False)
                
                if home_or_away_team1 == "home":
                    team1_box_score_list[i] = box_scores[boxscore_number_team1].home_score
                elif home_or_away_team1 == "away":
                    team1_box_score_list[i] = box_scores[boxscore_number_team1].away_score
                    
                if home_or_away_team2 == "home":
                    team2_box_score_list[i] = box_scores[boxscore_number_team2].home_score
                elif home_or_away_team2 == "away":
                    team2_box_score_list[i] = box_scores[boxscore_number_team2].away_score
            except (IndexError, AttributeError):
                print("box score out of range")
                # Handle errors gracefully if box score data is not available
                pass


    real_and_predicted_scores_team1 = predicted_values_from_present_team1
    real_and_predicted_scores_team2 = predicted_values_from_present_team2
    
    # Update Predicted Values with Box Scores
    for index, date in enumerate(dates):
        if date < today_minus_8:
            real_and_predicted_scores_team1[index] = team1_box_score_list[index]
            real_and_predicted_scores_team2[index] = team2_box_score_list[index]

        if index > 0:
            predicted_values_team1[index] += predicted_values_team1[index - 1]
            predicted_values_team2[index] += predicted_values_team2[index - 1]
            real_and_predicted_scores_team1[index] += real_and_predicted_scores_team1[index - 1]
            real_and_predicted_scores_team2[index] += real_and_predicted_scores_team2[index - 1]
    

    real_and_predicted_scores_team1.insert(0, 0)
    real_and_predicted_scores_team2.insert(0, 0)
    del real_and_predicted_scores_team1[-1]
    del real_and_predicted_scores_team2[-1]

    predicted_values_team1.insert(0, 0)
    predicted_values_team2.insert(0, 0)
    del predicted_values_team1[-1]
    del predicted_values_team2[-1]

    print(real_and_predicted_scores_team1)
    print(predicted_values_team1)
    #Convert Predicted Values into DataFrames
    team1_df = pd.DataFrame({
        'date': dates,
        'predicted_fpts': predicted_values_team1,
        'predicted_fpts_from_present': real_and_predicted_scores_team1,
        'team': 'Team 1'
    })

    team2_df = pd.DataFrame({
        'date': dates,
        'predicted_fpts': predicted_values_team2,
        'predicted_fpts_from_present': real_and_predicted_scores_team2,
        'team': 'Team 2'
    })

    combined_df = pd.concat([team1_df, team2_df], ignore_index=True)
    
    return combined_df

def get_compare_graphs_categories(league, team1_index, team1_player_data, team2_index, team2_player_data, year, week_data=None):
    
    # Use selected week data if provided, otherwise use current week
    if week_data and 'matchup_data' in week_data:
        # Use the selected week's date range and matchup period
        start_date = datetime.strptime(week_data['matchup_data']['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(week_data['matchup_data']['end_date'], '%Y-%m-%d').date()
        selected_matchup_period = week_data['selected_week']
    else:
        # Fall back to current week if no week_data provided
        today = datetime.today()
        today_minus_8 = (today-timedelta(hours=8)).date()
        start_date, end_date = db_utils.range_of_current_week(today_minus_8)
        selected_matchup_period = league.currentMatchupPeriod

    team1 = league.teams[team1_index]
    team2 = league.teams[team2_index]
    today = datetime.today()
    today_minus_8 = (today-timedelta(hours=8)).date()  

    # Create date range for the selected week
    dates = pd.date_range(start=start_date, end=end_date+timedelta(hours=24)).date
    print(start_date, end_date)

    boxscore_number_team1, home_or_away_team1 = get_team_boxscore_number(league, team1, selected_matchup_period)
    boxscore_number_team2, home_or_away_team2 = get_team_boxscore_number(league, team2, selected_matchup_period)

    # Get Weekly Box Scores for Both Teams
    team1_box_score_list = []
    team2_box_score_list = []

    # Get scoring periods from week_data if available
    if week_data and 'matchup_data' in week_data:
        scoring_periods = week_data['matchup_data']['scoring_periods']
    else:
        matchup_periods = db_utils.get_matchup_periods(league, selected_matchup_period)
        scoring_periods = []
        for matchup_period in matchup_periods:
            for i, date in enumerate(dates):
                scoring_periods.append(i + (matchup_period - 1) * 7)

    # Initialize box score lists with zeros
    team1_box_score_list = [0] * len(dates)
    team2_box_score_list = [0] * len(dates)

    # Get historical data for dates that have already passed
    for i, date in enumerate(dates):
        if date < today_minus_8:
            try:
                # Use scoring period from the matchup_data if available
                scoring_period = scoring_periods[i] if i < len(scoring_periods) else i + (selected_matchup_period - 1) * 7
                
                box_scores = league.box_scores(matchup_period=selected_matchup_period, 
                                              scoring_period=scoring_period, 
                                              matchup_total=False)
                
                if home_or_away_team1 == "home":
                    team1_box_score_list[i] = box_scores[boxscore_number_team1].home_stats
                elif home_or_away_team1 == "away":
                    team1_box_score_list[i] = box_scores[boxscore_number_team1].away_stats
                    
                if home_or_away_team2 == "home":
                    team2_box_score_list[i] = box_scores[boxscore_number_team2].home_stats
                elif home_or_away_team2 == "away":
                    team2_box_score_list[i] = box_scores[boxscore_number_team2].away_stats
            except (IndexError, AttributeError):
                # Handle errors gracefully if box score data is not available
                pass

    cats = ['FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS']
    # categories have different names when used in flask due to "to" being a keyword and number not being allowed
    category_mapping = {
        'FG%': 'fg%',
        'FT%': 'ft%',
        '3PM': 'threeptm',
        'REB': 'reb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        'TO': 'turno',
        'PTS': 'pts'
    }

    predicted_values_team1 = {}
    predicted_values_from_present_team1 = {}
    predicted_values_team2 = {}
    predicted_values_from_present_team2 = {}

    # grab the predictions for each cat, stored in predicted_values_from_present_team1, predicted_values_from_present_team2
    for cat in cats:
        mapped_cat = category_mapping[cat]
        predicted_values_team1[cat], predicted_values_from_present_team1[cat], predicted_values_team2[cat], predicted_values_from_present_team2[cat] = calculate_cat_predictions(
            dates, today_minus_8, team1, team2, team1_player_data, team2_player_data, mapped_cat
        )

    combined_category_dataframes = {}

    # adding the numbers up for display in the graph
    for cat in cats:
        
        # Update Predicted Values with Box Scores
        for index, date in enumerate(dates):
            if date < today_minus_8 and isinstance(team1_box_score_list[index], dict) and cat in team1_box_score_list[index]:
                predicted_values_from_present_team1[cat][index] = team1_box_score_list[index][cat]['value']
                predicted_values_from_present_team2[cat][index] = team2_box_score_list[index][cat]['value']

            if index > 0:
                predicted_values_team1[cat][index] += predicted_values_team1[cat][index - 1]
                predicted_values_team2[cat][index] += predicted_values_team2[cat][index - 1]
                predicted_values_from_present_team1[cat][index] += predicted_values_from_present_team1[cat][index - 1]
                predicted_values_from_present_team2[cat][index] += predicted_values_from_present_team2[cat][index - 1]
        

        predicted_values_from_present_team1[cat].insert(0, 0)
        predicted_values_from_present_team2[cat].insert(0, 0)
        del predicted_values_from_present_team1[cat][-1]
        del predicted_values_from_present_team2[cat][-1]

        predicted_values_team1[cat].insert(0, 0)
        predicted_values_team2[cat].insert(0, 0)
        del predicted_values_team1[cat][-1]
        del predicted_values_team2[cat][-1]

        # Create DataFrame for Team 1 and Team 2 for the current category
        team1_df = pd.DataFrame({
            'date': dates,
            'predicted_fpts': predicted_values_team1[cat],
            'predicted_fpts_from_present': predicted_values_from_present_team1[cat],
            'team': 'Team 1',
            'category': cat
        })

        team2_df = pd.DataFrame({
            'date': dates,
            'predicted_fpts': predicted_values_team2[cat],
            'predicted_fpts_from_present': predicted_values_from_present_team2[cat],
            'team': 'Team 2',
            'category': cat
        })

        # Combine both dataframes
        combined_df = pd.concat([team1_df, team2_df], ignore_index=True)

        # Store the combined DataFrame in the dictionary using the category as the key
        combined_category_dataframes[cat] = combined_df

    # Return the dictionary of DataFrames
    return combined_category_dataframes

def calculate_cat_predictions(dates, today_minus_8, team1, team2, team1_player_data, team2_player_data, mapped_cat):
    # Step 3: Initialize Dictionaries for Predicted Values
    dates_dict = {date: i for i, date in enumerate(dates)}
    predicted_values_team1 = [0] * len(dates)
    predicted_values_from_present_team1 = [0] * len(dates)
    predicted_values_team2 = [0] * len(dates)
    predicted_values_from_present_team2 = [0] * len(dates)

    def calculate_fpts_for_team(team, team_player_data, predicted_values, predicted_values_from_present, dates_dict):
        for player in team.roster:
            player_name = player.name
            player_row = team_player_data[team_player_data['player_name'] == player_name]

            if player_row.empty:
                avg_stat = 0
            else:
                avg_stat = player_row[mapped_cat].values[0]

            if not isinstance(avg_stat, str) and player_row['inj'].values[0] != 'OUT':
                list_schedule = list(player.schedule.values())
                list_schedule.sort(key=lambda x: x['date'], reverse=False)

                for game in list_schedule:  # timezone adjustment
                    game_date = (game['date'] - timedelta(hours=5)).date()
                    if game_date in dates_dict:
                        predicted_values[dates_dict[game_date]] += avg_stat
                        if game_date >= today_minus_8:
                            predicted_values_from_present[dates_dict[game_date]] += avg_stat

    # Calculate FPTS for both teams
    calculate_fpts_for_team(team1, team1_player_data, predicted_values_team1, predicted_values_from_present_team1, dates_dict)
    calculate_fpts_for_team(team2, team2_player_data, predicted_values_team2, predicted_values_from_present_team2, dates_dict)

    
    return predicted_values_team1, predicted_values_from_present_team1, predicted_values_team2, predicted_values_from_present_team2


def get_team_boxscore_number(league, team, matchup_period=None):

    for index, boxscore in enumerate(league.box_scores(matchup_period=matchup_period, matchup_total=False)):
        if team == boxscore.home_team:
            return index, "home"

        elif team == boxscore.away_team:  
            return index, "away"

    return "error"