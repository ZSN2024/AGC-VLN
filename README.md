# AGC-VLN: Air-Ground Collaborative Vision-and-Language Navigation

Training-free air-ground collaborative VLN via a **shared bird's-eye map**.
A UAV with a global bird's-eye view and a UGV with a local first-person view
cooperate to reach a target: the UAV runs **3D-SPF** (three-dimensional spatial
search) and renders the teammate's pose and the VLM-anchored target as
`CAR`/`GOAL` markers into the map, while the UGV plans a road path on that map
and drives it under closed-loop control. No training is required — both agents
query a frozen VLM (`gemini-3.7-flash`).

On 100 closed-loop episodes in CARLA-Air's Town10HD, AGC-VLN reaches a
**77.0%** joint success rate, a **+27.0%** collaboration gain over the weaker
single agent.

## Repository structure

```
AGC-VLN/
├── collab_UAV_UGV/        # main pipeline: UAV 3D-SPF + UGV VLM path planning
├── collab_UAV_UGV_alt/    # UAV initial-altitude ablation (30/60/90/120 m)
├── collab_UAV_UGV_annot/  # map-annotation-richness ablation
├── episodes/              # Town10HD episodes (50 spawn points) + goal photos
├── config.json            # VLM configuration (API key)
├── requirements.txt       # Python dependencies
├── third_party/           # airsim-python client (bundled)
├── AirSimConfig/          # AirSim settings
└── env_setup/             # CARLA-Air environment setup
```

## Setup

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. CARLA + AirSim environment (see env_setup/)
bash env_setup/setup_env.sh

# 3. VLM API key — copy the template and fill in your key
cp config.json.example config.json
export SPF_CCODE_API_KEY=...    # or set api_keys.ccode in config.json
```

`config.json.example` selects the VLM (`models.spf.model`, `base_urls`,
`api_keys`). The default model is `gemini-3.7-flash-c` over the `ccode` route.

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
