FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.11-slim

COPY --from=uv_bin /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Installation de libgomp1 (pour XGBoost / OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* requirements.txt* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f requirements.txt ]; then uv pip install --system -r requirements.txt; \
    else uv pip install --system -r pyproject.toml; fi

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]