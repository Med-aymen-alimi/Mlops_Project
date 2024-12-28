# Use an official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files into the container
COPY . .

# Expose port 8501 (this is the default port for Streamlit)
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "main.py", "--server.port", "8501"]