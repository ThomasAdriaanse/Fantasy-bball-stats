# Fantasy Basketball Stats - Setup Guide

## Prerequisites
- Docker installed on your system
- Database credentials (PostgreSQL)
- For private leagues: ESPN S2 and SWID values

## Quick Start with Docker

1. Build the Docker image:
```bash
docker build -t fantasy-scraper .
```

2. Run the container:
```bash
docker run -d -p 5000:5000 \
  -e DB_HOST=your_db_host \
  -e DB_NAME=your_db_name \
  -e DB_USER=your_db_user \
  -e DB_PASS=your_db_password \
  --name fantasy-scraper-app \
  fantasy-scraper
```

Replace the environment variables with your actual database credentials:
- `your_db_host`: Database host address
- `your_db_name`: Database name
- `your_db_user`: Database username
- `your_db_pass`: Database password

## Useful Docker Commands

- Check container status: `docker ps`
- View logs: `docker logs fantasy-scraper-app`
- Stop container: `docker stop fantasy-scraper-app`
- Start container: `docker start fantasy-scraper-app`
- Remove container: `docker rm fantasy-scraper-app`

## Accessing the Application

Once running, access the application at: http://localhost:5000

## Getting ESPN Credentials (for Private Leagues)

To access private leagues, you'll need ESPN S2 and SWID values:
1. Log into your ESPN Fantasy account
2. Open browser developer tools (F12)
3. Go to Application/Storage > Cookies
4. Find and copy the values for:
   - `espn_s2`
   - `SWID`

## Troubleshooting

If the container fails to start:
1. Check logs: `docker logs fantasy-scraper-app`
2. Verify database credentials
3. Ensure port 5000 is not in use by another application 