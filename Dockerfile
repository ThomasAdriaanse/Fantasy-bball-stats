# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory contents into the container at /usr/src/app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000


# Define environment variable for production
ENV FLASK_ENV=production

# Command to run your app
CMD ["python", "app.py"]
