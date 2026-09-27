FROM python:3.12-slim

# Copie des exécutables uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dependency système requise pour LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie des fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installation propre via le lockfile
RUN uv sync --frozen --no-cache

# Copie du reste du code
COPY . .

# Exécution directe via le venv créé par uv
CMD [".venv/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]