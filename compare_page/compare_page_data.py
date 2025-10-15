import pandas as pd
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

def get_team_player_data(
    league,
    team_num,
    columns,
    year,
    league_scoring_rules,
    week_data=None,
    stat_window: str | None = None,   # "projected"|"total"|"last_30"|"last_15"|"last_7" or None
):
    """
    Pull team player rows using the chosen ESPN window.

    Behavior:
      - If stat_window is provided -> STRICT mode (no fallbacks). If the window is invalid or
        has no 'avg' data, the row will be 'N/A'.
      - If stat_window is None -> FRIENDLY mode with fallbacks (projected -> total -> last_30 -> last_15 -> last_7).
    """
    from datetime import datetime, timedelta
    import pandas as pd

    team = league.teams[team_num]
    team_data = {column: [] for column in columns}

    VALID_WINDOWS = {"projected", "total", "last_30", "last_15", "last_7"}

    # === Helpers ===
    def pick(d, *keys, default=0):
        for k in keys:
            if k in d:
                return d[k]
        return default

    def r2(x):
        try:
            return round(float(x), 2)
        except Exception:
            return 0.0

    def pct(numer, denom):
        try:
            n = float(numer)
            d = float(denom)
            return round((n * 100.0) / d, 2) if d else 0.0
        except Exception:
            return 0.0

    def extract_avg(stats_block: dict | None) -> dict:
        if not stats_block or not isinstance(stats_block, dict):
            return {}
        if "avg" in stats_block and isinstance(stats_block["avg"], dict):
            return stats_block["avg"]
        return {}

    # Determine strict vs friendly mode
    strict_mode = stat_window is not None
    if strict_mode:
        window = (stat_window or "").strip().lower()
        # In strict mode: invalid window -> force empty (N/A), do NOT coerce to projected
        candidate_keys = [f"{year}_{window}"] if window in VALID_WINDOWS else []
    else:
        # Friendly mode default and fallbacks (like your old behavior)
        window = "projected"
        candidate_keys = [f"{year}_{window}", f"{year}_total", f"{year}_last_30", f"{year}_last_15", f"{year}_last_7"]

    for player in team.roster:
        stats_all = getattr(player, "stats", {}) or {}

        player_avg_stats = {}
        chosen_key = None

        # Only try the candidate keys we decided above
        for k in candidate_keys:
            if k in stats_all:
                player_avg_stats = extract_avg(stats_all.get(k))
                chosen_key = k
                if player_avg_stats:  # non-empty avg -> use it
                    break

        if player_avg_stats:
            # Pull with aliases (safe to handle ESPN names)
            MIN  = r2(pick(player_avg_stats, "MIN", "MPG", default=0))
            FGM  = r2(pick(player_avg_stats, "FGM", default=0))
            FGA  = r2(pick(player_avg_stats, "FGA", default=0))
            FTM  = r2(pick(player_avg_stats, "FTM", default=0))
            FTA  = r2(pick(player_avg_stats, "FTA", default=0))
            TPM  = r2(pick(player_avg_stats, "3PM", "TPM", default=0))
            REB  = r2(pick(player_avg_stats, "REB", "RPG", default=0))
            AST  = r2(pick(player_avg_stats, "AST", "APG", default=0))
            STL  = r2(pick(player_avg_stats, "STL", "SPG", default=0))
            BLK  = r2(pick(player_avg_stats, "BLK", "BPG", default=0))
            TOs  = r2(pick(player_avg_stats, "TO",  "TOPG", default=0))
            PTS  = r2(pick(player_avg_stats, "PTS", "PPG", default=0))

            FG_PCT = pct(FGM, FGA)
            FT_PCT = pct(FTM, FTA)

            # Fill row
            team_data['player_name'].append(player.name)
            team_data['min'].append(MIN)
            team_data['fgm'].append(FGM)
            team_data['fga'].append(FGA)
            team_data['fg%'].append(FG_PCT)
            team_data['ftm'].append(FTM)
            team_data['fta'].append(FTA)
            team_data['ft%'].append(FT_PCT)
            team_data['threeptm'].append(TPM)
            team_data['reb'].append(REB)
            team_data['ast'].append(AST)
            team_data['stl'].append(STL)
            team_data['blk'].append(BLK)
            team_data['turno'].append(TOs)
            team_data['pts'].append(PTS)
            team_data['inj'].append(getattr(player, "injuryStatus", None))

            fpts = round(
                FGM * league_scoring_rules.get('fgm', 0)
                + FGA * league_scoring_rules.get('fga', 0)
                + FTM * league_scoring_rules.get('ftm', 0)
                + FTA * league_scoring_rules.get('fta', 0)
                + TPM * league_scoring_rules.get('threeptm', 0)
                + REB * league_scoring_rules.get('reb', 0)
                + AST * league_scoring_rules.get('ast', 0)
                + STL * league_scoring_rules.get('stl', 0)
                + BLK * league_scoring_rules.get('blk', 0)
                + TOs * league_scoring_rules.get('turno', 0)
                + PTS * league_scoring_rules.get('pts', 0),
                2
            )
            team_data['fpts'].append(fpts)
        else:
            # Strict mode or no usable data -> N/A row
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
            team_data['inj'].append(getattr(player, "injuryStatus", None))
            team_data['fpts'].append('N/A')

        # === Games this week (unchanged) ===
        schedule = getattr(player, "schedule", {}) or {}
        list_schedule = list(schedule.values())
        list_schedule.sort(key=lambda x: x['date'], reverse=False)

        if week_data and 'matchup_data' in week_data and week_data['matchup_data']:
            start_date = datetime.strptime(week_data['matchup_data']['start_date'], '%Y-%m-%d').date()
            end_date   = datetime.strptime(week_data['matchup_data']['end_date'],   '%Y-%m-%d').date()
            games_in_week = [
                g for g in list_schedule
                if start_date <= (g['date'] - timedelta(hours=5)).date() <= end_date
            ]
        else:
            today_minus_8 = (datetime.today() - timedelta(hours=8)).date()
            start_of_week, end_of_week = range_of_current_week(today_minus_8)
            games_in_week = [
                g for g in list_schedule
                if today_minus_8 <= (g['date'] - timedelta(hours=5)).date() <= end_of_week
            ]

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
        start_date, end_date = range_of_current_week(today_minus_8)
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
        matchup_periods = get_matchup_periods(league, selected_matchup_period)
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
        start_date, end_date = range_of_current_week(today_minus_8)
        selected_matchup_period = league.currentMatchupPeriod

    team1 = league.teams[team1_index]
    team2 = league.teams[team2_index]
    today = datetime.today()
    today_minus_8 = (today-timedelta(hours=8)).date()  


    # Create date range for the selected week
    dates = pd.date_range(start=start_date, end=end_date+timedelta(hours=24)).date

    boxscore_number_team1, home_or_away_team1 = get_team_boxscore_number(league, team1, selected_matchup_period)
    boxscore_number_team2, home_or_away_team2 = get_team_boxscore_number(league, team2, selected_matchup_period)

    # Get Weekly Box Scores for Both Teams
    team1_box_score_list = []
    team2_box_score_list = []

    # Get scoring periods from week_data if available
    if week_data and 'matchup_data' in week_data:
        scoring_periods = week_data['matchup_data']['scoring_periods']
    else:
        matchup_periods = get_matchup_periods(league, selected_matchup_period)
        scoring_periods = []
        for matchup_period in matchup_periods:
            for i, date in enumerate(dates):
                scoring_periods.append(i + (matchup_period - 1) * 7)

    # Initialize box score lists with zeros
    team1_box_score_list = [0] * len(dates)
    team2_box_score_list = [0] * len(dates)

    # Get historical data for dates that have already passed
    def _sum_starter_raw(lineup):
        EXCLUDE_SLOTS = {"BE", "IL", ""}  # adjust if your league uses other non-scoring slots
        totals = {}
        for p in lineup or []:
            slot = getattr(p, "slot_position", getattr(p, "lineupSlot", ""))
            if slot in EXCLUDE_SLOTS:
                continue
            for k, v in (getattr(p, "points_breakdown", {}) or {}).items():
                totals[k] = totals.get(k, 0) + v
        return totals

    def _to_cat_values(raw_totals):
        # Compute FG% / FT% from makes/attempts if available; others are additive
        fgm = raw_totals.get("FGM", 0)
        fga = raw_totals.get("FGA", 0)
        ftm = raw_totals.get("FTM", 0)
        fta = raw_totals.get("FTA", 0)

        cat_vals = {
            "FG%": (fgm / fga) if fga else 0.0,
            "FT%": (ftm / fta) if fta else 0.0,
            "3PM": raw_totals.get("3PM", raw_totals.get("FG3M", 0)),
            "REB": raw_totals.get("REB", 0),
            "AST": raw_totals.get("AST", 0),
            "STL": raw_totals.get("STL", 0),
            "BLK": raw_totals.get("BLK", 0),
            "TO":  raw_totals.get("TO", 0),
            "PTS": raw_totals.get("PTS", 0),

            # >>> add base counters so the update loop can overwrite predictions with ACTUALS
            "FGM": fgm,
            "FGA": fga,
            "FTM": ftm,
            "FTA": fta,
        }
        # Wrap to match downstream access pattern: team_box_score_list[i][cat]['value']
        return {k: {"value": v} for k, v in cat_vals.items()}



    for i, date in enumerate(dates):
        if date < today_minus_8:
            try:
                # Use scoring period from the matchup_data if available
                scoring_period = scoring_periods[i] if i < len(scoring_periods) else i + (selected_matchup_period - 1) * 7

                boxes = league.box_scores(
                    matchup_period=selected_matchup_period,
                    scoring_period=scoring_period,
                    matchup_total=False
                )

                # pick the correct matchup for each team
                box_team1 = boxes[boxscore_number_team1]
                box_team2 = boxes[boxscore_number_team2]

                # choose home/away, sum starters only, convert to category values
                if home_or_away_team1 == "home":
                    raw1 = _sum_starter_raw(box_team1.home_lineup)
                else:
                    raw1 = _sum_starter_raw(box_team1.away_lineup)
                team1_box_score_list[i] = _to_cat_values(raw1)

                if home_or_away_team2 == "home":
                    raw2 = _sum_starter_raw(box_team2.home_lineup)
                else:
                    raw2 = _sum_starter_raw(box_team2.away_lineup)
                team2_box_score_list[i] = _to_cat_values(raw2)

            except (IndexError, AttributeError):
                # Handle errors gracefully if box score data is not available
                pass

    percentage_cats = []

    cats = ['3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS', 'FTM', 'FTA', 'FGM', 'FGA']
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
        'PTS': 'pts',
        'FTM': 'ftm',
        'FTA': 'fta',
        'FGM': 'fgm',
        'FGA': 'fga'
    }

    predicted_values_team1 = {}
    predicted_values_from_present_team1 = {}
    predicted_values_team2 = {}
    predicted_values_from_present_team2 = {}

    # grab the predictions for each cat, stored in predicted_values_from_present_team1, predicted_values_from_present_team2
    for cat in cats:
        mapped_cats = category_mapping[cat]
        predicted_values_team1[cat], predicted_values_from_present_team1[cat], predicted_values_team2[cat], predicted_values_from_present_team2[cat] = calculate_cat_predictions(
            dates, today_minus_8, team1, team2, team1_player_data, team2_player_data, mapped_cats
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
            'predicted_cat': predicted_values_team1[cat],
            'predicted_cat_from_present': predicted_values_from_present_team1[cat],
            'team': 'Team 1',
            'category': cat
        })

        team2_df = pd.DataFrame({
            'date': dates,
            'predicted_cat': predicted_values_team2[cat],
            'predicted_cat_from_present': predicted_values_from_present_team2[cat],
            'team': 'Team 2',
            'category': cat
        })

        # Combine both dataframes
        combined_df = pd.concat([team1_df, team2_df], ignore_index=True)

        # Store the combined DataFrame in the dictionary using the category as the key
        combined_category_dataframes[cat] = combined_df

    # --- Compute FG% / FT% from cumulative makes/attempts, set nice baselines, and put them FIRST ---
    def _ratio_list(num, den):
        return [(float(n) / float(d) if float(d) != 0 else 0.0) for n, d in zip(num, den)]

    # element-wise ratios from the (already cumulative) counters
    fgp1  = _ratio_list(predicted_values_team1['FGM'], predicted_values_team1['FGA'])
    fgp2  = _ratio_list(predicted_values_team2['FGM'], predicted_values_team2['FGA'])
    fgp1p = _ratio_list(predicted_values_from_present_team1['FGM'], predicted_values_from_present_team1['FGA'])
    fgp2p = _ratio_list(predicted_values_from_present_team2['FGM'], predicted_values_from_present_team2['FGA'])

    ftp1  = _ratio_list(predicted_values_team1['FTM'], predicted_values_team1['FTA'])
    ftp2  = _ratio_list(predicted_values_team2['FTM'], predicted_values_team2['FTA'])
    ftp1p = _ratio_list(predicted_values_from_present_team1['FTM'], predicted_values_from_present_team1['FTA'])
    ftp2p = _ratio_list(predicted_values_from_present_team2['FTM'], predicted_values_from_present_team2['FTA'])

    # set starting element baselines (FG% -> 0.47, FT% -> 0.78)
    FG_BASE, FT_BASE =  0.47, 0.78
    for seq in (fgp1, fgp2, fgp1p, fgp2p):
        if seq: seq[0] = FG_BASE
    for seq in (ftp1, ftp2, ftp1p, ftp2p):
        if seq: seq[0] = FT_BASE

    def _mk_df(cat, t1, t1p, t2, t2p):
        t1df = pd.DataFrame({'date': dates, 'predicted_cat': t1,  'predicted_cat_from_present': t1p,
                            'team': 'Team 1', 'category': cat})
        t2df = pd.DataFrame({'date': dates, 'predicted_cat': t2,  'predicted_cat_from_present': t2p,
                            'team': 'Team 2', 'category': cat})
        return pd.concat([t1df, t2df], ignore_index=True)

    fg_df = _mk_df('FG%', fgp1, fgp1p, fgp2, fgp2p)
    ft_df = _mk_df('FT%', ftp1, ftp1p, ftp2, ftp2p)

    # remove helper counters so they don't render as charts
    for k in ('FTM', 'FTA', 'FGM', 'FGA'):
        combined_category_dataframes.pop(k, None)

    # rebuild dict so FG%, FT% are first
    ordered_combined = {'FG%': fg_df, 'FT%': ft_df}
    for k in ['3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS']:
        if k in combined_category_dataframes:
            ordered_combined[k] = combined_category_dataframes[k]

    combined_category_dataframes = ordered_combined

    # Return the dictionary of DataFrames
    return combined_category_dataframes

def calculate_cat_predictions(dates, today_minus_8, team1, team2, team1_player_data, team2_player_data, mapped_cat):


    # Initialize Dictionaries for Predicted Values
    dates_dict = {date: i for i, date in enumerate(dates)}
    predicted_values_team1 = [0] * len(dates)
    predicted_values_from_present_team1 = [0] * len(dates)
    predicted_values_team2 = [0] * len(dates)
    predicted_values_from_present_team2 = [0] * len(dates)

    def calculate_cat_for_team(team, team_player_data, predicted_values, predicted_values_from_present, dates_dict):
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


    # Calculate cats for both teams
    calculate_cat_for_team(team1, team1_player_data, predicted_values_team1, predicted_values_from_present_team1, dates_dict)
    calculate_cat_for_team(team2, team2_player_data, predicted_values_team2, predicted_values_from_present_team2, dates_dict)

    return (
        [round(val, 2) for val in predicted_values_team1],
        [round(val, 2) for val in predicted_values_from_present_team1],
        [round(val, 2) for val in predicted_values_team2],
        [round(val, 2) for val in predicted_values_from_present_team2]
    )


def get_team_boxscore_number(league, team, matchup_period=None):

    for index, boxscore in enumerate(league.box_scores(matchup_period=matchup_period, matchup_total=False)):
        if team == boxscore.home_team:
            return index, "home"

        elif team == boxscore.away_team:  
            return index, "away"

    return "error"

def range_of_current_week(date):
    #today = datetime.today()
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week

def get_matchup_periods(league, current_matchup_period):
    return(league.settings.matchup_periods[str(current_matchup_period)])
