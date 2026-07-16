# Rebalancing Copilot — app container (Streamlit + engine).
# The local LLM (Ollama) runs as a SEPARATE service; see docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# non-root user; data dir writable for the SQLite db + log
RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# talk to the ollama service by its compose name; db/log persist on a volume
ENV OLLAMA_HOST=http://ollama:11434 \
    COPILOT_DB=/app/data/copilot.db \
    COPILOT_LOG=/app/data/copilot.log

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
