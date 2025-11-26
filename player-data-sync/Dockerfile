FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy sync script
COPY sync_player_data.py .

# Run the sync script
CMD ["python", "sync_player_data.py"]
