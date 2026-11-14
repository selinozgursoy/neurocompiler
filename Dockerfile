# One container serves the API and the web studio on port 7860.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
WORKDIR /app

COPY pyproject.toml README.md ./
COPY neurograph ./neurograph
COPY api ./api
COPY web ./web
RUN pip install .

# Hugging Face Spaces runs containers as a non-root user.
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health')" || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
