# Base image
FROM python:3.10-slim

# Work directory inside container
WORKDIR /app

# Install system deps (optional but good for numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Hugging Face uses
EXPOSE 7860

# Start FastAPI app (note: app.main:app from app/main.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
