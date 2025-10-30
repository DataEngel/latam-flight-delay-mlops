# ==============================
#   LATAM Delay Prediction API
#   Dockerfile (Local + Cloud Run)
# ==============================

# 1️⃣ Base image
FROM python:3.10.12-slim AS base

# 2️⃣ Working directory
WORKDIR /app

# 3️⃣ System dependencies
RUN apt-get update -y && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 4️⃣ Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy model and source code
COPY . .

# 6️⃣ Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 7️⃣ Expose API port
EXPOSE 8080

# 8️⃣ Default command for local run (same as Cloud Run)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
