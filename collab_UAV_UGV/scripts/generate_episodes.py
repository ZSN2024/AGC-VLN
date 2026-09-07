#!/usr/bin/env python3
"""Extract CARLA spawn points from a running map and generate episode templates.

Usage:
  # Terminal 1: start CARLA-Air
  ./carlaAir.sh Town03

  # Terminal 2: generate episode templates
  conda activate carlaAir
  python scripts/generate_episodes.py --map Town03 --num 15 --min-dist 80 --max-dist 300
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import carla


def connect(port: int = 2000, timeout: float = 20.0):
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(timeout)
    return client


def get_spawn_points(world) -> list[dict]:
    """Return all vehicle spawn points as dicts with x, y, z, yaw."""
    points = []
    for i, sp in enumerate(world.get_map().get_spawn_points()):
        loc = sp.location
        points.append({
            "index": i,
            "x": round(loc.x, 1),
            "y": round(loc.y, 1),
            "z": round(loc.z, 1),
            "yaw": round(sp.rotation.yaw, 1),
        })
    return points


def distance(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def sample_goal_near_spawn(
    spawn_points: list[dict],
    start: dict,
    min_dist: float,
    max_dist: float,
    rng: random.Random,
) -> dict | None:
    """Pick a goal spawn point within [min_dist, max_dist] of start."""
    candidates = [p for p in spawn_points if min_dist <= distance(p, start) <= max_dist]
    if not candidates:
        return None
    return rng.choice(candidates)


def sample_uav_spawn(start: dict, goal: dict, altitude: float) -> dict:
    """Place UAV centered above the midpoint of start-goal at given altitude."""
    mid_x = (start["x"] + goal["x"]) / 2
    mid_y = (start["y"] + goal["y"]) / 2
    return {"x": round(mid_x, 1), "y": round(mid_y, 1), "z": altitude, "yaw": 0.0}


def generate(args: argparse.Namespace) -> list[dict]:
    client = connect(args.carla_port)
    world = client.get_world()
    spawn_points = get_spawn_points(world)

    print(f"Map: {world.get_map().name}")
    print(f"Total spawn points: {len(spawn_points)}")

    rng = random.Random(args.seed)
    episodes = []
    used_pairs = set()

    attempts = 0
    max_attempts = args.num * 50

    while len(episodes) < args.num and attempts < max_attempts:
        attempts += 1
        ugv_idx = rng.randrange(len(spawn_points))
        ugv_spawn = spawn_points[ugv_idx]

        goal_spawn = sample_goal_near_spawn(
            spawn_points, ugv_spawn, args.min_dist, args.max_dist, rng
        )
        if goal_spawn is None:
            continue

        pair_key = (ugv_spawn["index"], goal_spawn["index"])
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        dist = distance(ugv_spawn, goal_spawn)
        uav_spawn = sample_uav_spawn(ugv_spawn, goal_spawn, args.uav_altitude)

        episodes.append({
            "id": f"{args.map.lower()}_{len(episodes)+1:03d}",
            "map": args.map,
            "instruction": f"[TODO: 写指令] 从 #{ugv_spawn['index']} 到 #{goal_spawn['index']} (距离 {dist:.0f}m)",
            "goal": {"x": goal_spawn["x"], "y": goal_spawn["y"], "z": goal_spawn["z"]},
            "ugv_spawn": {
                "index": ugv_spawn["index"],
                "x": ugv_spawn["x"],
                "y": ugv_spawn["y"],
                "z": ugv_spawn["z"],
                "yaw": ugv_spawn["yaw"],
            },
            "uav_spawn": uav_spawn,
            "distance_m": round(dist, 1),
            "time_budget_s": 180,
            "tags": [],
        })

    if len(episodes) < args.num:
        print(f"WARNING: only generated {len(episodes)}/{args.num} episodes "
              f"(try reducing --min-dist or increasing --max-dist)")

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Generate episode templates from running CARLA-Air")
    parser.add_argument("--map", default="Town03", help="Map name (must match running instance)")
    parser.add_argument("--num", type=int, default=15, help="Number of episodes to generate")
    parser.add_argument("--min-dist", type=float, default=80.0,
                        help="Minimum UGV start-to-goal distance (meters)")
    parser.add_argument("--max-dist", type=float, default=300.0,
                        help="Maximum UGV start-to-goal distance (meters)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--uav-altitude", type=float, default=50.0,
                        help="UAV flying altitude (meters)")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: episodes/<map>_templates.json)")
    args = parser.parse_args()

    episodes = generate(args)

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / "episodes"
    out_file = out_dir / f"{args.map.lower()}_templates.json"

    output = {
        "meta": {
            "map": args.map,
            "generated_by": "scripts/generate_episodes.py",
            "settings": {
                "min_dist_m": args.min_dist,
                "max_dist_m": args.max_dist,
                "uav_altitude_m": args.uav_altitude,
                "seed": args.seed,
            },
        },
        "episodes": episodes,
        "spawn_points": [],  # omit from template to keep file small
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(episodes)} episodes to {out_file}")
    print("")
    print("Next step: review each episode, write the 'instruction' field, add 'tags'.")

    # Also print spawn points summary for reference
    print(f"\nSpawn point count: {len(get_spawn_points(connect(args.carla_port).get_world()))}")


if __name__ == "__main__":
    main()
