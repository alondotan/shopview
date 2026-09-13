---
title: ShopView
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ShopView

Shopper analytics from store video: people detection and tracking (YOLO11 +
BoT-SORT), one stable id per visitor via clothing-based stitching, held-object
detection, camera-to-plan calibration, zones, a live-stream pipeline, and a
simulation viewer with a chat over the weekly summary.

* **Tabs:** Simulation · Video · Calibration · Live map · Zones · Live
* **Docs:** [vision/README.md](vision/README.md) — models, the floor-point
  estimate, why ids used to jump and what fixed it, the live pipeline, and
  how to deploy this container.
* **Run locally:** `pip install -r requirements.txt -r vision/requirements.txt`
  then `python server.py` (put `ANTHROPIC_API_KEY` in `.env` for the chat
  tab), or `docker build -t shopview . && docker run -p 7860:7860 shopview`.

The demo ships one sample video (a public CCTV clip), its analysis, the store
plan and the calibration. The **Live** tab runs YOLO on the host's CPU: on a
shared 2-vCPU host expect 2–4 analysed frames per second with held objects off.
