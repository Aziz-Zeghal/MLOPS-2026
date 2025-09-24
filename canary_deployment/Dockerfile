# Dockerfile for FastAPI MLflow Webservice

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code only (no trained model)
COPY . .

# Expose FastAPI port
EXPOSE 5000

# Entrypoint for FastAPI app
CMD ["fastapi", "run", "model_api.py", "--port", "5000"]
