FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1

COPY pyproject.toml README.md ./
COPY bip ./bip
COPY dbt ./dbt
RUN pip install --no-cache-dir ".[mlops,stream]"

EXPOSE 8000
CMD ["uvicorn", "bip.mlops.serve:app", "--host", "0.0.0.0", "--port", "8000"]
