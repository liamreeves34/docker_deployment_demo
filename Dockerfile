# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy only the requirements file first from the local app folder
COPY app/requirements.txt .

# Install dependencies without saving the massive cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else from the local app folder into the container's /app folder
COPY app/ .

# Expose the port FastAPI uses
EXPOSE 8000

# Run the API (Since main.py is now directly in the /app root, the command is just main:app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]