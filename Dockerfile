FROM python:3.11-slim

# Public demo defaults: adaptation (GPU training) is off, predictions run on this
# CPU host, and the worked example uses a pre-adapted model fetched at startup
# when FLOW_PREADAPTED_REPO is set (see docs/flow/DEMO.md).
ENV PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    HF_REPO=KrishMalik/deltav-fluid \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FLOW_DATA_DIR=/home/app/flow-data \
    FLOW_EXAMPLE_DIR=/home/app/flow-examples \
    FLOW_PREADAPTED_DIR=/home/app/preadapted \
    FLOW_ADAPT_ENABLED=0 \
    FLOW_EMBEDDED_WORKER=1

# Run as an unprivileged user (uid 1000, as Hugging Face Spaces expects).
RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH

WORKDIR /app

COPY requirements.txt .

# CPU-only torch wheel first (avoids pulling the multi-GB CUDA build);
# neuralop's own torch dependency is then already satisfied.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app main.py deltav_core.py ./
COPY --chown=app:app deltav_flow/ ./deltav_flow/
COPY --chown=app:app static/ ./static/
COPY --chown=app:app samples/ ./samples/

# Worked-example bundles and the sample study, downloaded byte for byte from the
# dataset at a pinned revision and checked against their SHA-256 (see the script).
COPY --chown=app:app scripts/fetch_site_files.py /tmp/build/
USER app
RUN python /tmp/build/fetch_site_files.py --examples "$FLOW_EXAMPLE_DIR" --downloads /app/static/downloads \
    && rm -rf "$HOME/.cache/huggingface"

EXPOSE 7860

# sh -c so ${PORT} expands (Render injects $PORT, HF Spaces expects 7860); exec keeps signals working.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
