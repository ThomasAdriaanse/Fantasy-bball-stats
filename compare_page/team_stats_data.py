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

def get_team_stats(league, team_num, team_player_data, opponent_num, opponent_player_data, columns, league_scoring_rules, year, week_data=None):

    # Use selected week matchup period if available
    selected_matchup_period = week_data['selected_week'] if week_data else league.currentMatchupPeriod

    #start_time = time.time()
    #print("Starting get_team_stats...")

    # Finding which team in the box scores dictionary we are
    team_boxscore_num = -1
    team_home_or_away = "temp"
    opponent_boxscore_num = -1
    opponent_home_or_away = "temp"

    box_count = 0
    for boxscore in league.box_scores(matchup_period=selected_matchup_period):
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

    #mid_time = time.time()
    #print(f"Time taken to find teams in box scores: {mid_time - start_time:.4f} seconds")

    team = league.teams[team_num]
    team_data = {column: [] for column in columns}
    opponent = league.teams[opponent_num]
    opponent_data = {column: [] for column in columns}
    
    # Calculating average fpts of the team    
    def get_team_avg_fpts(team, year):
        year_string = str(year) + "_total"
        team_average_fpts = 0
        num_players = 0

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

        return team_average_fpts

    team_data['team_avg_fpts'].append(get_team_avg_fpts(team, year))
    opponent_data['team_avg_fpts'].append(get_team_avg_fpts(opponent, year))

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

    current_scoring_period = league.scoringPeriodId
 

    # Get current box scores
    try:
        if team_home_or_away == "home":
            team_current_points = league.box_scores(matchup_period=selected_matchup_period)[team_boxscore_num].home_score
            try:
                current_period_scores = league.box_scores(matchup_period=selected_matchup_period, scoring_period=current_scoring_period, matchup_total=False)
                if team_boxscore_num < len(current_period_scores):
                    team_current_points -= current_period_scores[team_boxscore_num].home_score
            except (IndexError, AttributeError):
                print("error1")
                # If we can't get current period scores for future weeks, just use total points
                pass
        else:
            team_current_points = league.box_scores(matchup_period=selected_matchup_period)[team_boxscore_num].away_score
            try:
                current_period_scores = league.box_scores(matchup_period=selected_matchup_period, scoring_period=current_scoring_period, matchup_total=False)
                if team_boxscore_num < len(current_period_scores):
                    team_current_points -= current_period_scores[team_boxscore_num].away_score
            except (IndexError, AttributeError):
                print("error1")
                # If we can't get current period scores for future weeks, just use total points
                pass

        if opponent_home_or_away == "home":
            opponent_current_points = league.box_scores(matchup_period=selected_matchup_period)[opponent_boxscore_num].home_score
            try:
                current_period_scores = league.box_scores(matchup_period=selected_matchup_period, scoring_period=current_scoring_period, matchup_total=False)
                if opponent_boxscore_num < len(current_period_scores):
                    opponent_current_points -= current_period_scores[opponent_boxscore_num].home_score
            except (IndexError, AttributeError):
                print("error1")
                # If we can't get current period scores for future weeks, just use total points
                pass
        else:
            opponent_current_points = league.box_scores(matchup_period=selected_matchup_period)[opponent_boxscore_num].away_score
            try:
                current_period_scores = league.box_scores(matchup_period=selected_matchup_period, scoring_period=current_scoring_period, matchup_total=False)
                if opponent_boxscore_num < len(current_period_scores):
                    opponent_current_points -= current_period_scores[opponent_boxscore_num].away_score
            except (IndexError, AttributeError):
                print("error1")
                # If we can't get current period scores for future weeks, just use total points
                pass
    except IndexError:
        print("error2")
        # If we're looking at a future matchup period with no box scores yet, set current points to 0
        team_current_points = 0
        opponent_current_points = 0


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
        #print(z_score)
        #print(probability_team_wins)
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

    opponent_data['team_expected_points'].append(round(opponent_total_expected, 2))
    opponent_data['team_chance_of_winning'].append(round(100-probability_team_wins*100, 2))
    opponent_data['team_name'].append(opponent.team_name)
    opponent_data['team_current_points'].append(opponent_current_points)

    #end_time = time.time()
    #print(f"Total time for get_team_stats: {end_time - start_time:.4f} seconds")

    df = pd.DataFrame(team_data)
    opponent_df = pd.DataFrame(opponent_data)
    return df, opponent_df

def get_team_stats_categories(
    league,
    team_num,
    team_player_data,
    opponent_num,
    opponent_player_data,
    league_scoring_rules,
    year,
    week_data=None
):
    import pandas as pd

    team = league.teams[team_num]
    opponent = league.teams[opponent_num]

    # Use selected week matchup period if available
    selected_matchup_period = week_data['selected_week'] if week_data else league.currentMatchupPeriod

    # Locate box scores (only needed to fetch "current" category totals if your get_cat_stats uses them)
    try:
        team_boxscore_num, team_home_or_away = cpd.get_team_boxscore_number(league, team, selected_matchup_period)
        opponent_boxscore_num, opponent_home_or_away = cpd.get_team_boxscore_number(league, opponent, selected_matchup_period)

        if team_home_or_away == "home":
            team_current_stats = league.box_scores(matchup_period=selected_matchup_period)[team_boxscore_num].home_stats
        else:
            team_current_stats = league.box_scores(matchup_period=selected_matchup_period)[team_boxscore_num].away_stats

        if opponent_home_or_away == "home":
            opponent_current_stats = league.box_scores(matchup_period=selected_matchup_period)[opponent_boxscore_num].home_stats
        else:
            opponent_current_stats = league.box_scores(matchup_period=selected_matchup_period)[opponent_boxscore_num].away_stats
    except Exception:
        # For future weeks / missing box scores, fall back to empty dicts
        team_current_stats = {}
        opponent_current_stats = {}

    # Categories you project (left are your labels; right are ESPN stat keys)
    cats      = ['fg%', 'ft%', 'threeptm', 'reb', 'ast', 'stl', 'blk', 'turno', 'pts']
    espn_cats = ['FG%', 'FT%', '3PM',     'REB', 'AST', 'STL', 'BLK', 'TO',    'PTS']

    # Collect expected values and win % per category
    team_expected           = {}
    opponent_expected       = {}
    team_win_percentage     = {}
    opponent_win_percentage = {}

    for cat, espn_cat in zip(cats, espn_cats):
        texp, opexp, twin, opwin = get_cat_stats(
            cat, espn_cat,
            team_player_data, opponent_player_data,
            team_current_stats, opponent_current_stats
        )
        team_expected[cat]           = texp
        opponent_expected[cat]       = opexp
        team_win_percentage[cat]     = twin
        opponent_win_percentage[cat] = opwin

    # --- Add meta fields expected by the template ---
    # Use the same keys the points view returns so Jinja can read them.
    team_expected['team_name'] = team.team_name
    opponent_expected['team_name'] = opponent.team_name

    # Category page header shows "team_current_points"; there’s no single number here,
    # so provide an empty string (or 0 if you prefer).
    team_expected['team_current_points'] = ''
    opponent_expected['team_current_points'] = ''

    # Build 1-row DataFrames
    df           = pd.DataFrame([team_expected])
    opponent_df  = pd.DataFrame([opponent_expected])
    win_pct_df        = pd.DataFrame([team_win_percentage])
    opponent_pct_df   = pd.DataFrame([opponent_win_percentage])

    return df, opponent_df, win_pct_df, opponent_pct_df, team_current_stats, opponent_current_stats


def get_cat_stats(cat, espn_cat, team_player_data, opponent_player_data, team_current_stats, opponent_current_stats):
    # Filter out non-numeric values and players who are OUT
    team_player_data = team_player_data[(team_player_data['inj'] != 'OUT') & (team_player_data[cat] != 'N/A')].copy()
    opponent_player_data = opponent_player_data[(opponent_player_data['inj'] != 'OUT') & (opponent_player_data[cat] != 'N/A')].copy()
    
    # Convert numeric columns to float
    for df in [team_player_data, opponent_player_data]:
        for col in [cat, 'games']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=[cat, 'games'], inplace=True)

    if cat in ['fg%', 'ft%']:
        # Determine the correct stat keys
        made_stat = 'fgm' if cat == 'fg%' else 'ftm'
        attempt_stat = 'fga' if cat == 'fg%' else 'fta'

        # Convert makes and attempts columns to numeric
        for df in [team_player_data, opponent_player_data]:
            for col in [made_stat, attempt_stat]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=[made_stat, attempt_stat], inplace=True)

        # Calculate expected makes and attempts for team and opponent
        team_expected_makes = sum(team_player_data[made_stat] * team_player_data['games'])
        team_expected_attempts = sum(team_player_data[attempt_stat] * team_player_data['games'])
        opponent_expected_makes = sum(opponent_player_data[made_stat] * opponent_player_data['games'])
        opponent_expected_attempts = sum(opponent_player_data[attempt_stat] * opponent_player_data['games'])

        # Add current season stats
        team_current_makes = float(team_current_stats.get(made_stat, {}).get('value', 0))
        team_current_attempts = float(team_current_stats.get(attempt_stat, {}).get('value', 0))
        opponent_current_makes = float(opponent_current_stats.get(made_stat, {}).get('value', 0))
        opponent_current_attempts = float(opponent_current_stats.get(attempt_stat, {}).get('value', 0))

        # Compute final makes and attempts
        team_total_makes = team_current_makes + team_expected_makes
        team_total_attempts = team_current_attempts + team_expected_attempts
        opponent_total_makes = opponent_current_makes + opponent_expected_makes
        opponent_total_attempts = opponent_current_attempts + opponent_expected_attempts

        # Calculate percentages
        team_expected = (team_total_makes / team_total_attempts * 100) if team_total_attempts > 0 else 0
        opponent_expected = (opponent_total_makes / opponent_total_attempts * 100) if opponent_total_attempts > 0 else 0

        # For percentage stats, we need to consider the volume (attempts) in our variance calculation
        team_std = 2.0 * (team_total_attempts ** 0.5) if team_total_attempts > 0 else 0  # Using binomial variance approximation
        opponent_std = 2.0 * (opponent_total_attempts ** 0.5) if opponent_total_attempts > 0 else 0
        
        # Calculate z-score based on percentage difference and volume-adjusted standard deviation
        try:
            z_score = (team_expected - opponent_expected) / ((team_std**2 + opponent_std**2)**0.5)
            team_win_percentage = norm.cdf(z_score) * 100
        except ZeroDivisionError:
            team_win_percentage = 50.0 if team_expected == opponent_expected else (100.0 if team_expected > opponent_expected else 0.0)
            
    else:
        # Normal counting stat calculations
        team_current = float(team_current_stats.get(espn_cat, {}).get('value', 0))
        opponent_current = float(opponent_current_stats.get(espn_cat, {}).get('value', 0))
        
        team_expected = team_current + sum(team_player_data[cat] * team_player_data['games'])
        opponent_expected = opponent_current + sum(opponent_player_data[cat] * opponent_player_data['games'])

        # Adjust standard deviation based on stat type
        if cat == 'pts':
            std_factor = 0.25  # Points tend to be more consistent
        elif cat in ['threeptm', 'stl', 'blk']:
            std_factor = 0.60  # These stats tend to be more variable
        else:
            std_factor = 0.40  # Default variance for other stats

        # Calculate variance for each team
        team_variance = sum((team_player_data[cat] * std_factor * team_player_data['games'])**2)
        opponent_variance = sum((opponent_player_data[cat] * std_factor * opponent_player_data['games'])**2)
        
        # Calculate total standard deviation
        total_std = (team_variance + opponent_variance)**0.5
        
        # For turnovers, lower is better so invert the difference
        expected_difference = opponent_expected - team_expected if cat == 'turno' else team_expected - opponent_expected
        
        try:
            z_score = expected_difference / total_std if total_std > 0 else float('inf')
            team_win_percentage = norm.cdf(z_score) * 100
        except (ZeroDivisionError, ValueError):
            team_win_percentage = 50.0 if expected_difference == 0 else (100.0 if expected_difference > 0 else 0.0)

    opponent_win_percentage = 100 - team_win_percentage
    
    #print(f"Category: {cat}")
    #print(f"Team expected: {team_expected:.2f}")
    #print(f"Opponent expected: {opponent_expected:.2f}")
    #print(f"Team win %: {team_win_percentage:.2f}")
    #print(f"Opponent win %: {opponent_win_percentage:.2f}")


    
    return round(team_expected, 2), round(opponent_expected, 2), round(team_win_percentage, 2), round(opponent_win_percentage, 2)