FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt \
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

COPY src/ ./src/
COPY static/ ./static/
COPY models/ ./models/

EXPOSE 8080

CMD ["python", "-m", "src.server"]
