import pandas as pd
from scipy.stats import norm
import compare_page.compare_page_data as cpd
import time

def get_team_stats_data_schema():
    table_schema = """
        team_avg_fpts FLOAT,
        team_expected_points FLOAT,
        team_chance_of_winning FLOAT,
        team_name TEXT,
        team_current_points FLOAT
    """
    
    return table_schema

def get_team_stats(league, team_num, team_player_data, opponent_num, opponent_player_data, columns, league_scoring_rules, year):

    start_time = time.time()
    print("Starting get_team_stats...")

    # Finding which team in the box scores dictionary we are
    team_boxscore_num = -1
    team_home_or_away = "temp"
    opponent_boxscore_num = -1
    opponent_home_or_away = "temp"

    box_count = 0
    for boxscore in league.box_scores():
        if league.teams[team_num] == boxscore.home_team:
            team_boxscore_num = box_count 
            team_home_or_away = "home"

        elif league.teams[team_num] == boxscore.away_team:  
            team_boxscore_num = box_count 
            team_home_or_away = "away"

        if league.teams[opponent_num] == boxscore.home_team:
            opponent_boxscore_num = box_count 
            opponent_home_or_away = "home"

        elif league.teams[opponent_num] == boxscore.away_team:  
            opponent_boxscore_num = box_count 
            opponent_home_or_away = "away"
    
        box_count += 1

    if team_boxscore_num == -1 or opponent_boxscore_num == -1:
        print("Error: Could not find box scores for a team")

    mid_time = time.time()
    print(f"Time taken to find teams in box scores: {mid_time - start_time:.4f} seconds")

    team = league.teams[team_num]
    team_data = {column: [] for column in columns}

    # Get player in IR to exclude him from calculations
    if team_home_or_away == "home":
        lineup = league.box_scores()[team_boxscore_num].home_lineup
    else:
        lineup = league.box_scores()[team_boxscore_num].away_lineup

    # Calculating average fpts of the team
    team_average_fpts = 0
    num_players = 0

    year_string = str(year) + "_total"

    for player in team.roster:
        if year_string in player.stats and 'avg' in player.stats[year_string].keys():
            player_avg_stats = player.stats[year_string]['avg']

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

            team_average_fpts += fpts
            num_players += 1
        else:
            team_average_fpts += 0

    team_average_fpts /= num_players
    team_average_fpts = round(team_average_fpts, 2)

    team_data['team_avg_fpts'].append(team_average_fpts)

    # Filtering the DataFrame to get only players who are not 'out' and selecting their fantasy points
    player_averages = team_player_data[(team_player_data['inj'] != 'OUT') & (team_player_data['fpts'] != 'N/A')]['fpts'].tolist()
    player_games_left = team_player_data[(team_player_data['inj'] != 'OUT') & (team_player_data['fpts'] != 'N/A')]['games'].tolist()

    # Do the same for team 2
    opponent_player_averages = opponent_player_data[(opponent_player_data['inj'] != 'OUT') & (opponent_player_data['fpts'] != 'N/A')]['fpts'].tolist()
    opponent_player_games_left = opponent_player_data[(opponent_player_data['inj'] != 'OUT') & (opponent_player_data['fpts'] != 'N/A')]['games'].tolist()

    # Assuming the standard deviation is 40% of the average fantasy points for each player
    player_std_dev = 0.40
    team_1_stds = [avg * player_std_dev for avg in player_averages]
    team_2_stds = [avg * player_std_dev for avg in opponent_player_averages]

    # Calculate expected points remaining
    team_expected_points_remaining = sum(player_avg_fpts * games_left for player_avg_fpts, games_left in zip(player_averages, player_games_left))
    opponent_expected_points_remaining = sum(player_avg_fpts * games_left for player_avg_fpts, games_left in zip(opponent_player_averages, opponent_player_games_left))


    # Get current box scores
    if team_home_or_away == "home":
        team_current_points = league.box_scores()[team_boxscore_num].home_score
    else:
        team_current_points = league.box_scores()[team_boxscore_num].away_score

    if opponent_home_or_away == "home":
        opponent_current_points = league.box_scores()[opponent_boxscore_num].home_score
    else:
        opponent_current_points = league.box_scores()[opponent_boxscore_num].away_score

    team_total_expected = team_expected_points_remaining + team_current_points
    opponent_total_expected = opponent_expected_points_remaining + opponent_current_points

    # Calculate the total variance and standard deviation for each team
    total_variance_team_1 = sum((std * games_left)**2 for std, games_left in zip(team_1_stds, player_games_left))
    total_variance_team_2 = sum((std * games_left)**2 for std, games_left in zip(team_2_stds, opponent_player_games_left))
    total_std_team_1 = total_variance_team_1**0.5
    total_std_team_2 = total_variance_team_2**0.5
    expected_point_difference = team_total_expected - opponent_total_expected

    try:
        z_score = expected_point_difference / ((total_std_team_2**2 + total_std_team_1**2)**0.5)
        probability_team_wins = norm.cdf(z_score)
        print(z_score)
        print(probability_team_wins)
    except ZeroDivisionError:            
        z_score = float('inf')
        if expected_point_difference > 0:
            probability_team_wins = 1.0  
        else:
            probability_team_wins = 0.0
        
        if expected_point_difference == 0:
            probability_team_wins = 0.5

    team_data['team_expected_points'].append(round(team_total_expected, 2))
    team_data['team_chance_of_winning'].append(round(probability_team_wins*100, 2))
    team_data['team_name'].append(team.team_name)
    team_data['team_current_points'].append(team_current_points)

    end_time = time.time()
    print(f"Total time for get_team_stats: {end_time - start_time:.4f} seconds")

    df = pd.DataFrame(team_data)
    return df
