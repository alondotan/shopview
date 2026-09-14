# ShopView — simulation viewer + video analytics (YOLO) in one container.
# Built for CPU hosts such as Hugging Face Spaces (port 7860, runs as uid 1000),
# Railway or Render (they set $PORT). ~2 GB image: torch CPU + ultralytics.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/yolo OMP_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# non-root user with uid 1000 (what Hugging Face Spaces runs containers as)
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt ./req/requirements.txt
COPY vision/requirements.txt ./req/vision-requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r req/requirements.txt -r req/vision-requirements.txt

COPY --chown=user:user . .

# The sample videos live in Git LFS. Hosts that clone without LFS (Railway,
# Render) hand us the 3-line pointer instead; fetch the real files from GitHub.
RUN python -c "import os, glob, urllib.request; \
    [os.path.getsize(f) > 10000 or urllib.request.urlretrieve('https://github.com/alondotan/shopview/raw/master/' + f, f) \
     for f in sorted(glob.glob('data/videos/*.mp4'))]; \
    print({f: os.path.getsize(f) // 1000000 for f in sorted(glob.glob('data/videos/*.mp4'))}, 'MB')"

# fetch the model weights at build time so the first request is not a download:
# person/pose detector + YOLO-World into data/models (where detect_people looks),
# and YOLO-World's CLIP text encoder, which ultralytics keeps under ./weights
# relative to the working directory — so it is fetched from /app, the runtime cwd.
RUN mkdir -p data/models data/live && chown -R user:user /app
USER user
ENV HOME=/home/user
RUN cd data/models && python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt'); YOLO('yolov8s-worldv2.pt')"
RUN python -c "from ultralytics import YOLOWorld; m = YOLOWorld('data/models/yolov8s-worldv2.pt'); m.set_classes(['shopping bag'])"

EXPOSE 7860
# one process (the live pipeline lives in it), many threads (SSE streams hold connections)
CMD gunicorn server:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --worker-class gthread --threads 16 --timeout 0 --access-logfile -
