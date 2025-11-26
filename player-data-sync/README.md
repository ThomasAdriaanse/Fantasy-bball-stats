# Player Data Sync Service

Standalone Docker service that syncs NBA player data from S3 to a shared Docker volume.

## Quick Start

### 1. Build the image:
```bash
docker build -t player-data-sync .
```

### 2. Run the sync:
```bash
docker run --rm \
  -v player_data:/app/data/players \
  -e S3_BUCKET=fantasy-stats-dev \
  -e S3_PREFIX=dev/players/ \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  player-data-sync
```

## Integration with Main App

Add this to your main app's `docker-compose.yml`:

```yaml
services:
  web:
    # Your existing web service
    volumes:
      - player_data:/app/data/players:ro  # Read-only access
    environment:
      - PLAYER_DATA_CACHE_DIR=/app/data/players

  sync:
    build: ./player-data-sync
    volumes:
      - player_data:/app/data/players  # Read-write access
    environment:
      - S3_BUCKET=${S3_BUCKET}
      - S3_PREFIX=${S3_PREFIX:-dev/players/}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_REGION=${AWS_REGION:-ca-central-1}
    profiles:
      - sync  # Only runs when explicitly called

volumes:
  player_data:  # Shared volume
```

## Running the Sync

### Manual run:
```bash
docker-compose run --rm sync
```

### Weekly cron job (on EC2):
```bash
# Add to crontab (crontab -e):
0 2 * * 0 cd /path/to/Fantasy-Scraper-website && docker-compose run --rm sync >> /var/log/player_sync.log 2>&1
```

### First-time setup:
```bash
# Run sync to populate the volume
docker-compose run --rm sync

# Then start your web app
docker-compose up -d web
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_BUCKET` | Yes | - | S3 bucket name |
| `S3_PREFIX` | No | `dev/players/` | S3 prefix for player data |
| `PLAYER_DATA_CACHE_DIR` | No | `/app/data/players` | Local cache directory |
| `AWS_REGION` | No | `ca-central-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | Yes | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | - | AWS secret key |

## How It Works

1. **Sync service** downloads all player JSON files from S3 to `/app/data/players` in the Docker volume
2. **Web app** reads from the same volume (read-only) for instant access
3. **Docker volume** persists data between container restarts
4. **Weekly cron** keeps data fresh

## Volume Location

The Docker volume `player_data` is stored on your EC2 instance at:
```
/var/lib/docker/volumes/fantasy-scraper-website_player_data/_data/
```

You can inspect it with:
```bash
docker volume inspect fantasy-scraper-website_player_data
```

## Troubleshooting

### Check if data exists:
```bash
docker run --rm -v player_data:/data alpine ls -lh /data
```

### View sync logs:
```bash
docker-compose run --rm sync
```

### Clear and re-sync:
```bash
docker volume rm fantasy-scraper-website_player_data
docker-compose run --rm sync
```
