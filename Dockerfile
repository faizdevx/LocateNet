# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
ARG INSTALL_VIDEO=0
ENV PIP_NO_CACHE_DIR=1
# Compiler is needed in case a dependency (for example insightface) has to build from source.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements.txt requirements-video.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_VIDEO" = "1" ]; then pip install -r requirements-video.txt; fi

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LOCATENET_DB=/app/data/database.db
# Shared libraries OpenCV, FAISS and onnxruntime need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app . .
RUN mkdir -p data resources uploads logs vector_db/storage /home/app/.insightface \
    && chown -R app:app /app /home/app
USER app
EXPOSE 5000
# One worker: the face models are memory-heavy and the FAISS index lives in this process.
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]
