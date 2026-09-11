FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    HF_REPO=KrishMalik/deltav-fluid

COPY requirements.txt .

# CPU-only torch wheel first (avoids pulling the multi-GB CUDA build);
# neuralop's own torch dependency is then already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py deltav_core.py ./
COPY static/ ./static/
COPY samples/ ./samples/

EXPOSE 7860

# Shell form so ${PORT} expands: Render injects $PORT, HF Spaces expects 7860.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
