#!/usr/bin/env python3
"""Aggregate a collab_UAV_UGV_annot batch into Table-4 metrics.

Usage:
  python collab_UAV_UGV_annot/scripts/summarize_batch.py <BATCH_DIR>

Scans <BATCH_DIR>/collab_<episode>_*/result.json (15 scenes x N runs) for ONE
annotation-richness level, computes SR_UGV / SR_joint (union) / CG / Time, and
writes results.csv + summary.txt into <BATCH_DIR>. Run once per level and
collect the per-level lines into Table 4.

Note: SR_joint uses the union of uav_success and ugv_success (NOT the
`success` field, which is the AND of the two).
"""

import csv
import json
import math
import re
import sys
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt_mean_std(xs):
    if not xs:
        return "N/A"
    m = mean(xs)
    if len(xs) >= 2:
        return f"{m:.1f}±{std(xs):.1f}"
    return f"{m:.1f}"


def main():
    batch = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    results = sorted(batch.glob("collab_*/result.json"))
    if not results:
        print(f"No result.json found under {batch}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for r in results:
        d = json.loads(r.read_text(encoding="utf-8"))
        m = re.search(r"collab_(town10hd_\d+)_", r.parent.name)
        ep = m.group(1) if m else r.parent.name
        us = bool(d.get("uav_success"))
        gs = bool(d.get("ugv_success"))
        rows.append({
            "ep": ep,
            "run": r.parent.name,
            "uav_success": us,
            "ugv_success": gs,
            "joint": us or gs,
            "time_s": d.get("time_s"),
            "uav_dist": d.get("uav_final_dist_m"),
            "ugv_dist": d.get("ugv_final_dist_m"),
            "vlm": d.get("total_vlm_calls"),
        })

    n = len(rows)
    n_uav = sum(r["uav_success"] for r in rows)
    n_ugv = sum(r["ugv_success"] for r in rows)
    n_joint = sum(r["joint"] for r in rows)

    sr_uav = 100.0 * n_uav / n
    sr_ugv = 100.0 * n_ugv / n
    sr_joint = 100.0 * n_joint / n
    cg = sr_joint - max(sr_uav, sr_ugv)

    joint_times = [r["time_s"] for r in rows if r["joint"] and r["time_s"] is not None]
    all_times = [r["time_s"] for r in rows if r["time_s"] is not None]

    scenes = {}
    for r in rows:
        scenes.setdefault(r["ep"], []).append(r)
    scene_lines = []
    for ep in sorted(scenes):
        rs = scenes[ep]
        m = len(rs)
        s_u = sum(x["uav_success"] for x in rs)
        s_g = sum(x["ugv_success"] for x in rs)
        s_j = sum(x["joint"] for x in rs)
        scene_lines.append(f"  {ep}: {m} runs  UAV {s_u}/{m}  UGV {s_g}/{m}  joint {s_j}/{m}")

    summary = (
        "============================================\n"
        f"  Table-4 Summary ({batch.name})\n"
        f"  Runs: {n}\n"
        f"  SR_UAV:   {sr_uav:.1f}%  ({n_uav}/{n})\n"
        f"  SR_UGV:   {sr_ugv:.1f}%  ({n_ugv}/{n})\n"
        f"  SR_joint: {sr_joint:.1f}%  ({n_joint}/{n})  [union]\n"
        f"  CG:       {cg:.1f}%\n"
        f"  Time (joint succ): {fmt_mean_std(joint_times)} s\n"
        f"  Time (all):        {fmt_mean_std(all_times)} s\n"
        "--------------------------------------------\n"
        + "\n".join(scene_lines)
        + "\n============================================"
    )

    print(summary)

    csv_path = batch / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep", "run", "uav_success", "ugv_success", "joint",
                    "time_s", "uav_final_dist_m", "ugv_final_dist_m", "total_vlm_calls"])
        for r in rows:
            w.writerow([r["ep"], r["run"], r["uav_success"], r["ugv_success"],
                        r["joint"], r["time_s"], r["uav_dist"], r["ugv_dist"], r["vlm"]])

    (batch / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\nWrote {csv_path} and {batch / 'summary.txt'}")


if __name__ == "__main__":
    main()
