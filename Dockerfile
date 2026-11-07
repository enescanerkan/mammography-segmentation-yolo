# Positioning review app, CPU only.
#   docker build -t mammography-segmentation:cpu .
#   docker run -p 127.0.0.1:8000:8000 -v /path/to/weights:/weights:ro -v positioning-data:/data mammography-segmentation:cpu
FROM python:3.11-slim

# OpenCV (pulled in by ultralytics) needs libGL / glib even headless.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU wheels of torch first, so ultralytics does not pull the CUDA build (several GB). constraints.txt pins the
# library versions the measurements were validated with.
COPY constraints.txt ./
RUN pip install -c constraints.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install -c constraints.txt ".[app,models,dicom]" \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /data /weights && chown app /data

USER app
ENV BREAST_SEG_WEIGHTS=/weights
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["breast-seg", "serve", "--config", "configs/app.example.yaml", "--host", "0.0.0.0", "--port", "8000", \
     "--device", "cpu", "--data-dir", "/data"]
