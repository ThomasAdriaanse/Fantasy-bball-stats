
import unittest
import os
import sys
from unittest.mock import patch

# Setup paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'player-data-sync'))

os.environ["SESSION_KEY"] = "mock_key"
os.environ["S3_BUCKET"] = "mock-bucket"

from app import create_app

class TestRankingsRoute(unittest.TestCase):
    def setUp(self):
        # Create app in testing mode
        self.app = create_app()
        self.app.testing = True
        self.client = self.app.test_client()

    @patch('app.services.darko_services.get_darko_z_scores')
    def test_rankings_page_loads(self, mock_get_stats):
        # Mock data return
        mock_get_stats.return_value = [
            {
                "player_name": "Test Player",
                "team": "Test Team",
                "RAW_DARKO": {},
                "RAW_REAL": {},
                "Z_DARKO": {
                   "Z_PTS": 1.0, "Z_FG3M": 1.0, "Z_REB": 1.0,
                   "Z_AST": 1.0, "Z_STL": 1.0, "Z_BLK": 1.0,
                   "Z_TOV": 1.0, "Z_FG": 1.0, "Z_FT": 1.0
                },
                "Z_REAL": {
                   "Z_PTS": 0.5, "Z_FG3M": 0.5, "Z_REB": 0.5,
                   "Z_AST": 0.5, "Z_STL": 0.5, "Z_BLK": 0.5,
                   "Z_TOV": 0.5, "Z_FG": 0.5, "Z_FT": 0.5
                }
            }
        ]
        
        response = self.client.get('/rankings/rankings')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Player Rankings", response.data)
        self.assertIn(b"Test Player", response.data)
        self.assertIn(b"9.00", response.data) # Total Z sum

if __name__ == "__main__":
    unittest.main()
