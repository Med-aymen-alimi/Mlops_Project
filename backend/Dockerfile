FROM python:3.10-slim

# Define the working directory
WORKDIR /app

# Copy the .env file (optional)
COPY .env .env

# Copy the dependencies file and install packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the necessary files into the Docker image
COPY main.py main.py
COPY ./example_json ./example_json
COPY ./src ./src

# Expose the port used by the backend
EXPOSE 8081

# Command to run the application in production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]