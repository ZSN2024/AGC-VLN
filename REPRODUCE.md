# AGC-VLN Reproduction Guide

This document explains how to reproduce the simulation results of AGC-VLN
(Air-Ground Collaborative Vision-and-Language Navigation via Shared Bird's-Eye
Maps), in particular the **joint success = 77.0%** reported over 100
closed-loop episodes.

## 1. Environment

- **Simulator**: CARLA-Air (a single Unreal Engine process merging CARLA and
  AirSim), map **Town10HD**. CARLA port `2000`, AirSim port `41451`.
- **Python**: conda env `carlaAir` (Python 3.10).
- **VLM**: `config.json` → `models.spf`. Default model `gemini-3.7-flash-c`
  (ccode route, API key via env var `SPF_CCODE_API_KEY`). If `model` starts
  with `qwen`, the dashscope route is used (`SPF_DASHSCOPE_API_KEY`).

```bash
# Terminal 1: start the simulator
./CarlaAir.sh Town10HD

# Terminal 2: activate the env and set the API key
conda activate carlaAir
export SPF_CCODE_API_KEY=<your ccode key>
```

## 2. Benchmark dataset

`episodes/` contains:

| File | Content |
|---|---|
| `town10hd_templates.json` | 50 episode templates (start/goal spawn points, distance) |
| `town10hd_annotated.json` | 15 annotated episodes (with full English instructions) |
| `photos/town10hd_001_goal.jpg` … `015_goal.jpg` | 15 goal photos |

Each episode has fields: `id`, `map`, `instruction`, `goal{x,y,z}`,
`ugv_spawn{index,x,y,z,yaw}`, `uav_spawn{x,y,z,yaw}`, `distance_m`,
`time_budget_s` (180), `tags`.

> **About "100 episodes"**: the paper's 100 closed-loop episodes are the
> 50 scenes above, each run **twice** (100 runs total).

## 3. Episode generation

`collab_UAV_UGV/scripts/generate_episodes.py` generates the templates. It
requires CARLA-Air to be running (it reads `get_spawn_points()` live). The
50 templates were generated with:

```bash
python collab_UAV_UGV/scripts/generate_episodes.py \
    --map Town10HD --num 50 --min-dist 40 --max-dist 60 \
    --uav-altitude 50 --seed 42
```

## 4. Running the collaboration

```bash
# run one episode N times
bash collab_UAV_UGV/scripts/auto_run_collab.sh 3 town10hd_001

# equivalent single run
python collab_UAV_UGV/scripts/run_collab.py \
    --episode-id town10hd_001 --auto --time-limit 180 --run-dir <out>
```

Each run writes `result.json`, `summary.txt`, `log.txt`, and per-step
annotated/birdview images + VLM-call JSON under
`<run-dir>/collab_<episode>_<ts>/`.

Aggregate a batch:

```bash
python collab_UAV_UGV/scripts/summarize_batch.py <batch-dir>
```

## 5. Metrics

- **Success threshold**: `SUCCESS_DIST = 5.0` m (`run_collab.py` line 30).
- `uav_success` / `ugv_success`: either agent within 5 m of the goal.
- **joint success** = `uav_success OR ugv_success` (union).
- **CG** = `SR_joint − min(SR_uav, SR_ugv)`.

## 6. The 77.0% result

The 100 runs (50 scenes × 2) are provided separately as
`AGC-VLN_100episodes_batch_20260828_192843.zip` (per-step logs + images). The
aggregate metrics are:

| Metric | Value |
|---|---|
| joint success (union) | **77.0%** |
| UAV success | 50.0% |
| UGV success | 75.0% |
| CG | +27.0% |

The aggregate `results.csv` and `summary.txt` are kept in
[`results/`](results/).

## 7. Notes on the released code (code-paper alignment)

For transparency, here is how the released code relates to the paper's
description:

1. **Target anchoring**: the paper's method section describes the target as
   "anchored by a frozen VLM"; in the released code the red `GOAL` marker on
   the shared bird's-eye map is drawn from the episode's **ground-truth goal
   coordinate**, while the VLM (`vlm_uav`) localizes the target only for the
   UAV's own flight (3D-SPF).
2. **Instruction**: the 50 templates' `instruction` field is a `[TODO: ...]`
   placeholder (only the 15 annotated episodes have real English instructions),
   and `vlm_ugv`'s prompt does **not** inject the instruction text.
3. **Goal photos**: `photos/` has 15 goal photos, but the episode JSON has no
   `goal_photo` field pointing to them, so `ep.get("goal_photo")` returns
   `None` and the photos are not wired into the main flow.
4. **Success threshold**: `SUCCESS_DIST = 5.0` m, matching the paper's 5 m
   threshold (an earlier commit used 10.0).
