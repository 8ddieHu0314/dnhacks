FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

COPY services ./services
COPY docs/components ./docs/components

ENV PYTHONPATH=/app/services
CMD ["uvicorn", "vision_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
