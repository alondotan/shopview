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

Shopper analytics from store video: people detection and tracking (YOLOX +
RTMPose + ByteTrack — Apache-2.0 / MIT, no AGPL), one stable id per visitor via
clothing-based stitching, held-object detection, camera-to-plan calibration, zones, several cameras on one store
fused into one path per shopper, and a Live map tab that answers the same four
questions the simulation viewer does — map, analytics, paths and a chat — about
the people a camera actually recorded: dwell, occupancy, checkout waits and the
routes they walked. Plus a live-stream pipeline and the original simulation
viewer.

* **Tabs:** Simulation · Video · Calibration · Live map · Zones · Multi-cam calibration · Multi-cam · Live
* **Docs:** [vision/README.md](vision/README.md) — models and their licences,
  the floor-point estimate, why ids used to jump and what fixed it, how the
  statistics are counted (and what censoring does to them), the live pipeline,
  and how to deploy this container.
* **Run locally:** `pip install -r requirements.txt -r vision/requirements.txt`
  then `python server.py`, or `docker build -t shopview . && docker run -p
  7860:7860 shopview`. The chat tabs need a Claude credential — sign in with
  `ant auth login` (no key to manage), or put `ANTHROPIC_API_KEY` in `.env`.
  Everything else runs without one.

The demo ships one sample video (a public CCTV clip), its analysis, the store
plan and the calibration. The **Live** tab runs the detector on the host's
CPU: on a shared 2-vCPU host expect 2–4 analysed frames per second.
