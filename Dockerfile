FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    HF_REPO=KrishMalik/deltav-fluid \
    HF_HUB_DISABLE_TELEMETRY=1

# Run as an unprivileged user (uid 1000, as Hugging Face Spaces expects).
RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH

WORKDIR /app

COPY requirements.txt .

# CPU-only torch wheel first (avoids pulling the multi-GB CUDA build);
# neuralop's own torch dependency is then already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app main.py deltav_core.py ./
COPY --chown=app:app static/ ./static/
COPY --chown=app:app samples/ ./samples/

USER app

EXPOSE 7860

# Shell form so ${PORT} expands: Render injects $PORT, HF Spaces expects 7860.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
