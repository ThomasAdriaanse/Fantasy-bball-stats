from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague
from datetime import date, timedelta


def matchup_dates(league, league_year):
    today = date.today()
    if league_year < 2026:
        season_start = date(2025, 4, 13) - timedelta(days=league.scoringPeriodId)
    else:
        season_start = date(2025, 10, 20) if today < date(2025, 10, 20) else today - timedelta(days=league.scoringPeriodId)

    mp = league.settings.matchup_periods
    first = league.firstScoringPeriod
    out, prev_end, mult = {}, None, 0
    for mp_num, days in mp.items():
        mp_num = int(mp_num)
        length = len(days) + (1 if mp_num    == (18 - first) else 0)
        start = season_start if mp_num == 1 else prev_end + timedelta(days=1)
        end = start + timedelta(days=6 + 7*(length-1))
        scoring = [i + (mp_num - 1 + mult) * 7 for i in range((end - start).days + 1)]
        out[f"matchup_{mp_num}"] = {
            "matchup_period": mp_num,
            "scoring_periods": scoring,
            "start_date": start.strftime('%Y-%m-%d'),
            "end_date": end.strftime('%Y-%m-%d'),
        }
        mult += length - 1
        prev_end = end
    return out
