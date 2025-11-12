import os
from datetime import timedelta

class Config:
    PERMANENT_SESSION_LIFETIME = timedelta(days=31)
    S3_BUCKET = os.getenv("S3_BUCKET", "fantasy-stats-dev")
    S3_PREFIX = os.getenv("S3_PREFIX", "dev/players/")
    AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
    SECRET_KEY = os.environ["SESSION_KEY"]

class Production(Config):
    DEBUG = False
    SECRET_KEY = os.environ["SESSION_KEY"]

class Development(Config):
    DEBUG = True
    SECRET_KEY = os.environ.get("SESSION_KEY", "dev-not-secret-change-me")