
import os
import json
import csv
import sys
from pathlib import Path
from io import StringIO
import shutil

# Set up dummy environment for the test
# Use an absolute path for test dir to avoid confusion
TEST_DIR = Path(os.getcwd()) / "test_darko_fix"
if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)
TEST_DIR.mkdir()

DARKO_DIR = TEST_DIR / "darko"
PACE_DIR = TEST_DIR / "pace"
AVGS_DIR = TEST_DIR / "avgs"

DARKO_DIR.mkdir()
PACE_DIR.mkdir()
AVGS_DIR.mkdir()

# 1. Create dummy Pace file
pace_data = {
    "data": [
        {"TEAM_NAME": "Denver Nuggets", "PACE": 98.5},
        {"TEAM_NAME": "Oklahoma City Thunder", "PACE": 100.2}
    ]
}
with open(PACE_DIR / "team_pace.json", "w") as f:
    json.dump(pace_data, f)

# 2. Create dummy Season Averages file
# Keys must be slugs!
avgs_data = {
    "data": {
        "nikola_jokic": {"MIN": 34.5},
        "shai_gilgeous_alexander": {"MIN": 35.2}
    }
}
with open(AVGS_DIR / "season_averages.json", "w") as f:
    json.dump(avgs_data, f)

# 3. Create dummy DARKO CSV
csv_path = DARKO_DIR / "DARKO_player_talent_2025-01-01.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Player", "Team", "FGA/100", "FG3A/100", "FG3%", "FTA/100", "FT%", "REB/100", 
        "AST/100", "BLK/100", "STL/100", "TOV/100", "FG2%"
    ])
    writer.writerow([
        "Nikola Jokic", "Denver Nuggets", 20, 5, 0.4, 8, 0.85, 15, 12, 1, 1.5, 3, 0.6
    ])
    writer.writerow([
        "Shai Gilgeous-Alexander", "Oklahoma City Thunder", 22, 4, 0.38, 9, 0.9, 6, 7, 1, 2.5, 2.5, 0.55
    ])
    writer.writerow([
        "Mystery Player", "Denver Nuggets", 10, 2, 0.3, 2, 0.7, 3, 2, 0.5, 0.5, 1, 0.4
    ])

print("Setup complete. Running test...")

# Add current directory to path so we can import 'app'
sys.path.append(os.getcwd())

try:
    from app.services import darko_services
    
    # Override constants to point to our test dirs
    darko_services.DARKO_CACHE_DIR = str(DARKO_DIR)
    darko_services.TEAM_PACE_CACHE_DIR = str(PACE_DIR)
    darko_services.SEASON_AVGS_CACHE_DIR = str(AVGS_DIR)
    
    # Capture stdout
    captured_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output
    
    try:
        results = darko_services.get_raw_darko_stats()
    finally:
        sys.stdout = original_stdout
        
    output = captured_output.getvalue()
    print("Function output captured.")
    
    # Check results
    print(f"Found {len(results)} players.")
    
    # Check logs
    if "[DARKO] Missing player MPG for Nikola Jokic" in output:
        print("FAIL: Found fake missing log for Jokic")
    else:
        print("PASS: No missing log for Jokic")
        
    if "[DARKO] Missing player MPG for Shai Gilgeous-Alexander" in output:
        print("FAIL: Found fake missing log for SGA")
    else:
        print("PASS: No missing log for SGA")

    if "[DARKO] Missing player MPG for Mystery Player" in output:
        print("PASS: Correctly found missing log for Mystery Player")
    else:
        print("FAIL: Did NOT find missing log for Mystery Player")
        print("Output was:", output)

except Exception as e:
    print(f"CRASH: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)
