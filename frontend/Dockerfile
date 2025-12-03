FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app
CMD ["streamlit", "run", "frontend/app_streamlit.py", "--server.port", "8501", "--server.headless", "true"]
