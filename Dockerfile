FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

COPY log/ /app/log/

COPY run.sh /app/run.sh


RUN mkdir -p /app/data
RUN mkdir -p /app/output
RUN chmod +x /app/src/*.py || true
RUN chmod +x /app/run.sh || true

CMD ["bash", "/app/run.sh"]

