# ShopView — simulation viewer + video analytics (YOLOX + RTMPose + ByteTrack,
# all permissively licensed) in one container. Built for CPU hosts such as
# Hugging Face Spaces (port 7860, runs as uid 1000), Railway or Render (they set
# $PORT). ~700 MB image: onnxruntime + opencv, no torch. The optional OWLv2
# object detector / CLIP embedder (vision/requirements-owl.txt) are not
# installed — they need torch and are too slow for a 2-vCPU host anyway.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# non-root user with uid 1000 (what Hugging Face Spaces runs containers as)
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt ./req/requirements.txt
COPY vision/requirements.txt ./req/vision-requirements.txt
RUN pip install -r req/requirements.txt -r req/vision-requirements.txt

COPY --chown=user:user . .

# The sample videos live in Git LFS. Hosts that clone without LFS (Railway,
# Render) hand us the 3-line pointer instead; fetch the real files from GitHub.
RUN python -c "import os, glob, urllib.request; \
    [os.path.getsize(f) > 10000 or urllib.request.urlretrieve('https://github.com/alondotan/shopview/raw/master/' + f, f) \
     for f in sorted(glob.glob('data/videos/*.mp4'))]; \
    print({f: os.path.getsize(f) // 1000000 for f in sorted(glob.glob('data/videos/*.mp4'))}, 'MB')"

# fetch the model weights at build time so the first request is not a download:
# the default detector + pose model into data/models (where detector.py looks)
RUN mkdir -p data/models data/live && chown -R user:user /app
USER user
ENV HOME=/home/user
RUN python vision/detector.py --fetch

EXPOSE 7860
# one process (the live pipeline lives in it), many threads (SSE streams hold connections)
CMD gunicorn server:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --worker-class gthread --threads 16 --timeout 0 --access-logfile -
