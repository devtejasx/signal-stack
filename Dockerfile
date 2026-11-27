FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY signalstack ./signalstack
COPY scripts ./scripts
COPY demo ./demo
COPY deploy ./deploy
RUN mkdir -p data
EXPOSE 8000
# The weekly-brief and monthly-audit schedulers run inside this process; missed runs are caught up on start-up.
CMD ["sh", "deploy/start.sh"]
