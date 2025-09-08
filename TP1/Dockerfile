# Use official Python image as base
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model file
COPY src/regression.joblib ./

EXPOSE 9000

CMD ["fastapi", "run", "src/model_fastapi.py",  "--port", "9000"]