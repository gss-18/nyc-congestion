# Forecast API image. Expects the trained model and recent data to be mounted at /app/runs and /app/data:
#   docker build -t nyc-forecast .
#   docker run -p 8000:8000 -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data" nyc-forecast
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV NYC_CONGESTION_ROOT=/app
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "nyc_congestion.api:app", "--host", "0.0.0.0", "--port", "8000"]
