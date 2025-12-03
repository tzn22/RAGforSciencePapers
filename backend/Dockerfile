FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
