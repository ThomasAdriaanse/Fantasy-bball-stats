# app/services/z_score_calculations.py
from typing import Dict
import math

'''
input
avg_raw_for_z = {
            "PTS":  raw_pts,
            "FG3M": raw_3pm,
            "REB":  raw_reb,
            "AST":  raw_ast,
            "STL":  raw_stl,
            "BLK":  raw_blk,
            "TOV":  raw_to,
            "FGM":  fgm,
            "FGA":  fga,
            "FTM":  ftm,
            "FTA":  fta,
        }'''


def raw_to_zscore(raw_stats: Dict[str, float]) -> Dict[str, float]:
    # formulas taken from "Improving Algorithms for Fantasy Basketball" v2, Zach Rosenof, https://arxiv.org/pdf/2307.02188v2

    z_scores = {
        "Z_PTS": 0,
        "Z_FG3M": 0,
        "Z_REB": 0,
        "Z_AST": 0,
        "Z_STL": 0,
        "Z_BLK": 0,
        "Z_TOV": 0,
        "Z_FG": 0,
        "Z_FT": 0,
    }

    #means and std devs for top 156 players
    league_means: Dict[str, float] = {
        "PTS": 16.69286,
        "FG3M": 1.791283,
        "REB": 5.940254,
        "AST": 3.81490,
        "STL": 1.083269,
        "BLK": 0.702051,
        "TOV": 1.912358,
        "FGM": 6.094920,
        "FGA": 12.70326,
        "FTM": 2.711740,
        "FTA": 3.378334,    
    }

    league_std_devs: Dict[str, float] = {
        "PTS": 6.033309,
        "FG3M": 1.026308,
        "REB": 2.536052,
        "AST": 2.077686,
        "STL": 0.385049,
        "BLK": 0.526237,
        "TOV": 0.860546
        #these are not needed for the formula
        #"FGM": ,
        #"FGA": ,
        #"FTM": ,
        #"FTA": ,
    }

    average_success_rate_fg = 0.479791741
    average_success_rate_ft = 0.802685748
    std_dev_fg_formula = 0.056499321
    std_dev_ft_formula = 0.219905549

    # % cats formula:

    # (attempts made by player/Mean of attempts across players)*(player success rate - success rate across players)
    # /
    # Standard deviation of the above value across all players

    # counting cats formula:

    # (mean stat for player - mean stat across all players)/std_dev of stat across all players
    # numerator reversed for TOs


    cats = ['PTS', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FT', 'FG']


    for cat in cats:
        z_score_name = f"Z_{cat}"

        if cat == 'FT':
            if math.isnan(raw_stats['FTA']):
                z_scores[z_score_name] = 'NaN'
                continue
        elif cat == 'FG':
            if math.isnan(raw_stats['FGA']):
                z_scores[z_score_name] = 'NaN'
                continue
        elif math.isnan(raw_stats[cat]):
            z_scores[z_score_name] = 'NaN'
            continue

        if cat == 'FG':
            if raw_stats['FGA'] != 0:
                player_success_rate_fg =  raw_stats['FGM']/raw_stats['FGA']
                z_scores[z_score_name] = ((raw_stats['FGA']/league_means['FGA'])*(player_success_rate_fg-average_success_rate_fg))/std_dev_fg_formula
            else:
                z_scores[z_score_name] = 0
        elif cat =='FT':
            if raw_stats['FTA'] != 0:
                player_success_rate_ft =  raw_stats['FTM']/raw_stats['FTA']
                z_scores[z_score_name] = ((raw_stats['FTA']/league_means['FTA'])*(player_success_rate_ft-average_success_rate_ft))/std_dev_ft_formula
            else:
                z_scores[z_score_name] = 0
        elif cat == 'TOV':
            z_scores[z_score_name] =  (league_means[cat]-raw_stats[cat])/league_std_devs[cat]
        else:
            z_scores[z_score_name] =  (raw_stats[cat]-league_means[cat])/league_std_devs[cat]

    return z_scores
    