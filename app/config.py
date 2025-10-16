import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-not-secret-change-me")
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    S3_BUCKET = os.getenv("S3_BUCKET", "fantasy-stats-dev")
    S3_PREFIX = os.getenv("S3_PREFIX", "dev/players/")
    AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

class Production(Config):
    DEBUG = False

class Development(Config):
    DEBUG = True
