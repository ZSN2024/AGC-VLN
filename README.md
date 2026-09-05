<div align="center">

# AGC-VLN

### Air-Ground Collaborative Vision-and-Language Navigation via Shared Bird's-Eye Maps

**Shuning Zhang**<sup>1,\*</sup>, **Liang Li**<sup>2,\*</sup>, **Yunheng Wang**<sup>1</sup>, **Tao Wang**<sup>1</sup>, **Yihang Kang**<sup>1</sup>, **Renjing Xu**<sup>1</sup>

<sup>1</sup> Robotics and Autonomous Systems Thrust, Systems Hub, The Hong Kong University of Science and Technology (Guangzhou), China<br>
<sup>2</sup> School of Communications and Information Engineering, Nanjing University of Posts and Telecommunications, China<br>
<sup>\*</sup> Equal contribution

**Training-free air-ground collaboration — the UAV turns its "can see" into the UGV's "can drive".**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

</div>

## 🎥 Demo Video

<p align="center">
  <a href="assets/AGC-VLN.mp4">
    <img src="assets/video_poster.png" alt="AGC-VLN demo video" width="95%">
  </a>
  <br>
  <em>▶ <a href="assets/AGC-VLN.mp4">Watch the demo video</a> (2 min, MP4)</em>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="AGC-VLN teaser" width="95%">
</p>

## Overview

Air-ground collaborative **Vision-and-Language Navigation (VLN)** pairs an
unmanned aerial vehicle (UAV) with a global bird's-eye view and an unmanned
ground vehicle (UGV) with a local first-person view. Existing training-free
methods solve single-agent tasks but offer no collaboration mechanism, and
trained VLA models fail to turn single-agent skill into cooperative behavior.

**AGC-VLN** is the first *training-free* baseline for air-ground collaborative
VLN. The key insight is that training-free methods decompose navigation into
VLM-based *semantic reasoning* and deterministic *geometric execution*,
exposing a natural collaboration interface: a **shared bird's-eye map**.

- The **UAV** renders the UGV's reported pose and the VLM-anchored target as
  `CAR` / `GOAL` markers with distance labels onto its global bird's-eye view.
- The **UGV** reads this map, plans a road-following path with a frozen VLM,
  and executes it under closed-loop control.
- In parallel, the UAV runs **3D-SPF**, a spatial-search upgrade of
  See-Point-Fly (SPF) that localizes the target in the downward view and flies
  toward it.

No training, no learned cross-agent representation — just a frozen VLM
(`gemini-3.7-flash`) and deterministic geometry.

## Highlights

- 🚁 **3D-SPF** — adds a discrete height command (descend / hold / ascend) to
  SPF, turning fixed-altitude planar point-fly into three-dimensional spatial
  search (UAV success 20.0% → 60.0%, a 3× gain).
- 🗺️ **Shared bird's-eye map** — the UAV's global view fills the UGV's
  first-person blind spot, lifting the UGV from 12.0% → 75.0%.
- 🤝 **Positive collaboration gain** — joint success **77.0%**, a **+27.0%**
  gain over the weaker single agent, and **24.0 pts** above the best published
  single-agent baseline (Travel UAV, 53.0%).
- 🧊 **Training-free** — no gradients ever flow; a frozen VLM is orchestrated
  by deterministic geometry.
- 🤖 **Real-robot verified** — the same pipeline runs on a physical quadrotor
  and omnidirectional robot.

## Method

<p align="center">
  <img src="assets/overview.png" alt="AGC-VLN architecture" width="100%">
</p>

### UAV — global perception & map rendering (3D-SPF)

The UAV faces a downward RGB camera and, at each decision step (≈3 s, ~0.33 Hz):

1. **Anchors** the target in the downward image with a frozen VLM against the
   target photo `P_g`.
2. **Renders** a shared bird's-eye map: a blue `CAR` marker (the UGV pose), a
   red `GOAL` marker (the target), a green cross (the UAV itself), a yellow
   `CAR→GOAL` reference line, and distance labels.
3. **Runs 3D-SPF** for itself: the VLM returns the target's pixel position and
   a discrete height command `h ∈ {descend, hold, ascend}`. The pixel is
   projected to world coordinates via flat-ground ray casting, the UAV flies
   toward it with proportional velocity, then adjusts altitude accordingly.

### UGV — map path planning & closed-loop execution

The UGV carries a forward camera and, upon receiving the annotated map:

1. **Plans** a 10-waypoint road path with a frozen VLM (first point = its own
   position, last point = the target, intermediate points follow the road).
2. **Projects** the pixel waypoints back to world coordinates using the UAV's
   pose at annotation time.
3. **Tracks** the path point-by-point with a closed-loop controller that
   discretizes the heading error into six driving primitives (forward / left /
   right / reverse / reverse-left / reverse-right), with stuck recovery.

### Collaboration — view complementarity

The two agents exchange only two things through shared memory: the UGV reports
its pose, and the UAV shares the annotated bird's-eye map. The UAV sees globally
but cannot drive; the UGV can drive but sees only locally. The shared map
delivers the UAV's global spatial context to the UGV in an image form its VLM
can consume directly — filling the UGV's blind spot.

## Results

On 100 closed-loop episodes in CARLA-Air's Town10HD scene, AGC-VLN reaches a
**77.0% joint success rate** — a **+27.0% collaboration gain** over the weaker
individual agent — and exceeds the strongest published single-agent baseline
(Travel UAV, 53.0%) by **24.0 points**.

### Main results

| Method | SR_UGV ↑ | SR_UAV ↑ | SR_joint ↑ | SPL ↑ | NE (m) ↓ | CG ↑ | Time (s) ↓ | VLM calls ↓ | Path_UAV (m) ↓ | Path_UGV (m) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenFly | — | 0.0% | 0.0% | 0.0% | 78.4±18.7 | 0.0% | --- | 34.1±7.3 | 56.4±16.0 | — |
| FineCog-Nav | — | 13.0% | 13.0% | 10.0% | 28.4±12.9 | 0.0% | 177.2±35.4 | 58.0±12.4 | 19.9±10.4 | — |
| 3DG-VLN | — | 33.0% | 33.0% | 26.0% | 49.6±32.1 | 0.0% | 142.1±60.4 | 11.7±5.2 | 49.2±21.7 | — |
| Travel UAV | — | 53.0% | 53.0% | 45.0% | 30.3±24.3 | 0.0% | 115.8±63.4 | 21.0±11.5 | 42.24±27.70 | — |
| UGV-only VLM | 12.0% | — | 12.0% | 11.6% | 115.2±65.6 | 0.0% | 28.2±3.0 | 12.1±4.0 | — | 49.4±4.0 |
| UAV-only SPF | — | 20.0% | 20.0% | 18.0% | 155.5±94.9 | 0.0% | 58.5±9.6 | 12.9±6.5 | 119.1±74.1 | — |
| UAV-only 3D-SPF | — | 60.0% | 60.0% | 46.0% | 30.0±40.9 | 0.0% | 104.0±8.0 | 13.0±7.1 | 84.3±36.7 | — |
| **AGC-VLN (ours)** | **75.0%** | **50.0%** | **77.0%** | **62.0%** | **10.9±16.2** | **+27.0%** | **82.2±27.0** | **12.1±2.9** | **65.1±22.2** | **43.5±14.0** |

SR_UGV / SR_UAV / SR_joint = UGV / UAV / joint success rate (goal within 5 m).
SPL = success weighted by path length; NE = navigation error; CG = collaboration
gain = SR_joint − min(SR_UAV, SR_UGV); "—" = not applicable. The first block is
published single-agent baselines, the second our single-agent ablations, and the
last our full AGC-VLN system.

<p align="center">
  <img src="assets/ablation.png" alt="Ablation results" width="100%">
</p>

We ablate three design axes:

- **VLM backbone** — `gemini-3.7-flash` attains the highest joint success rate.
- **UAV initial altitude** — 60 m is optimal; too low narrows the field of view,
  too high shrinks the target to a handful of pixels.
- **Map annotation richness** — full annotation (CAR+GOAL+distance+line) is
  necessary; degrading it stepwise reduces the UGV's road-planning ability.

## Real-robot deployment

Beyond simulation, the same training-free pipeline runs on real hardware — a
quadrotor (Mid360 LiDAR + RealSense D435i) and an omnidirectional mobile
robot, both on Linux — confirming feasibility decoupled from the simulator's
synchronized pose stream.

<p align="center">
  <img src="assets/real_robot.png" alt="Real-robot deployment" width="60%">
</p>

## Repository structure

```
AGC-VLN/
├── collab_UAV_UGV/        # main pipeline: UAV 3D-SPF + UGV VLM path planning
├── collab_UAV_UGV_alt/    # UAV initial-altitude ablation (30/60/90/120 m)
├── collab_UAV_UGV_annot/  # map-annotation-richness ablation
├── episodes/              # Town10HD episodes (50 spawn points) + goal photos
├── assets/                # figures used in this README
├── config.json.example    # VLM configuration template (copy to config.json)
├── requirements.txt       # Python dependencies
├── third_party/           # airsim-python client (bundled)
├── AirSimConfig/          # AirSim settings
└── env_setup/             # CARLA-Air environment setup
```

## Installation

> **Prerequisite — CARLA-Air.** This code runs on top of
> [CARLA-Air](https://github.com/louiszengCN/CarlaAir)
> ([arXiv:2603.28032](https://arxiv.org/abs/2603.28032)), which merges CARLA and
> AirSim into a single Unreal Engine process. Set it up first following its own
> instructions, then continue with the steps below.

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. CARLA + AirSim environment (see env_setup/)
bash env_setup/setup_env.sh

# 3. VLM API key — copy the template and fill in your key
cp config.json.example config.json
export SPF_CCODE_API_KEY=...    # or set api_keys.ccode in config.json
```

`config.json` selects the VLM (`models.spf.model`, `base_urls`, `api_keys`).
The default model is `gemini-3.7-flash-c` over the `ccode` route.

## Usage

Run the main pipeline on one episode:

```bash
python collab_UAV_UGV/scripts/run_collab.py --episode-id town10hd_001 --auto
```

The UAV and UGV run as two parallel threads communicating through shared
memory. Each decision step (≈3 s) the UAV annotates the bird's-eye map and runs
3D-SPF, while the UGV plans a 10-waypoint road path and follows it with a
closed-loop controller. Success means either agent reaches the target within
10 m inside the time budget.

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2026agcvln,
  title   = {Air-Ground Collaborative Vision-and-Language Navigation via Shared Bird's-Eye Maps},
  author  = {Shuning Zhang and Liang Li and Yunheng Wang and Tao Wang and Yihang Kang and Renjing Xu},
  year    = {2026},
}
```

## License

This project is released under the [MIT License](LICENSE).
