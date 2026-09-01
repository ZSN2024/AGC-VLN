#!/usr/bin/env python3
"""Test car movement: verify the car can drive to given CARLA target positions.

Usage:
  conda activate carlaAir
  python collab_UAV_UGV/scripts/test_car_move.py                    # default test targets
  python collab_UAV_UGV/scripts/test_car_move.py --target 100,-50   # custom target
"""

import argparse, math, sys, time
import carla
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Helpers (same logic as debug_ugv.py / run_collab.py)
# ═══════════════════════════════════════════════════════════════

def get_car_state(car):
    """Return CARLA [x, y, z], yaw (rad), speed (m/s)."""
    loc = car.get_location()
    vel = car.get_velocity()
    yaw = math.radians(car.get_transform().rotation.yaw)
    return np.array([loc.x, loc.y, loc.z]), yaw, math.sqrt(vel.x ** 2 + vel.y ** 2)


def _classify_action(angle_err):
    """Classify angle error into driving primitive. Returns (action_name, steer, reverse)."""
    if abs(angle_err) < 30:
        return "forward", 0.0, False
    if abs(angle_err) > 150:
        return "reverse", 0.0, True
    if angle_err > 0:
        if angle_err > 120:
            return "reverse-right", -0.6, True
        return "forward-right", 0.6, False
    else:
        if angle_err < -120:
            return "reverse-left", 0.6, True
        return "forward-left", -0.6, False


def follow_path(car, world_waypoints, verbose=True, waypoint_radius=2.0, max_time_per_wp=15.0):
    """
    Closed-loop control: drive car through CARLA waypoints.
    Continuously reads position, adjusts throttle/steer, stops when close enough.
    """
    STEP = 0.1  # control loop interval (seconds)
    STALL_THRESHOLD = 2.0   # seconds of no movement before triggering unstuck

    for idx, wp in enumerate(world_waypoints):
        wp_xy = wp[:2]

        if verbose:
            print(f"  → wp{idx+1}/{len(world_waypoints)}: target=({wp_xy[0]:.0f},{wp_xy[1]:.0f})")

        t_start_wall = time.monotonic()
        t_start_sim = car.get_world().get_snapshot().timestamp.elapsed_seconds
        last_action = None
        stall_t = 0.0          # accumulated stall time
        last_pos = None

        while True:
            pos, yaw, spd = get_car_state(car)
            dx = wp_xy[0] - pos[0]
            dy = wp_xy[1] - pos[1]
            dist = float(np.linalg.norm([dx, dy]))

            # Arrived at this waypoint
            if dist < waypoint_radius:
                if verbose:
                    print(f"       ✓ arrived  dist={dist:.1f}m  speed={spd:.1f}m/s")
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.2)
                break

            # Timeout — use sim-time so lag doesn't cause premature timeout
            sim_elapsed = car.get_world().get_snapshot().timestamp.elapsed_seconds - t_start_sim
            if sim_elapsed > max_time_per_wp:
                wall_elapsed = time.monotonic() - t_start_wall
                ratio = sim_elapsed / wall_elapsed if wall_elapsed > 0 else 1.0
                if verbose:
                    print(f"       ⚠ timeout  dist={dist:.1f}m  "
                          f"wall={wall_elapsed:.1f}s sim={sim_elapsed:.1f}s ratio={ratio:.2f}x  moving on")
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.2)
                break

            # ── Stall detection: throttle engaged but car barely moving ──
            if last_pos is not None:
                moved = float(np.linalg.norm(pos[:2] - last_pos[:2]))
                if spd < 0.15 and moved < 0.02:
                    stall_t += STEP
                else:
                    stall_t = max(0.0, stall_t - STEP * 2)  # fast decay
            last_pos = pos.copy()

            if stall_t > STALL_THRESHOLD:
                if verbose:
                    print(f"       ⚡ STUCK {stall_t:.1f}s — reversing 5m to unstick")
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.3)
                # Reverse ~5m
                car.apply_control(carla.VehicleControl(throttle=0.8, steer=0.0, reverse=True))
                time.sleep(2.0)
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.3)
                stall_t = 0.0
                last_pos = None
                continue

            # ── Compute control ──
            target_yaw = math.atan2(dy, dx)
            angle_err = math.degrees((target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
            action, steer, reverse = _classify_action(angle_err)

            # Throttle proportional to distance, capped
            throttle = min(1.0, max(0.3, dist / 10.0))
            if reverse:
                throttle = min(1.0, max(0.3, dist / 10.0))

            # Apply control
            ctrl = carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse)
            car.apply_control(ctrl)

            if action != last_action:
                if verbose:
                    print(f"       dist={dist:.1f}m  angle_err={angle_err:+.0f}°  "
                          f"action={action:>14s}  throttle={throttle:.2f}  steer={steer:+.1f}")
                last_action = action

            time.sleep(STEP)

    car.apply_control(carla.VehicleControl(brake=1.0))
    time.sleep(0.3)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test car movement to target positions")
    parser.add_argument("--target", type=str, default=None,
                        help="Target CARLA x,y (e.g. '100,-50'). Without this, runs default tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    # ── Connect ──
    cl = carla.Client(args.host, args.port)
    cl.set_timeout(10.0)
    world = cl.get_world()
    print(f"Connected to CARLA {args.host}:{args.port}")

    # ── Find the Mini Cooper (UGV) ──
    start_car = None
    for v in world.get_actors().filter("vehicle.*"):
        tid = v.type_id.lower()
        if "carlacola" in tid or "hgv" in tid:
            continue
        if "mini" in tid:
            start_car = v
            break
    if start_car is None:
        print("ERROR: Mini Cooper not found — run setup_scene.py first")
        return False

    # Read initial state
    pos0, yaw0, spd0 = get_car_state(start_car)
    print(f"Car: {start_car.type_id}")
    print(f"Start position:  ({pos0[0]:.1f}, {pos0[1]:.1f}, {pos0[2]:.1f})")
    print(f"Start yaw:       {math.degrees(yaw0):.1f}°")
    print(f"Start speed:     {spd0:.1f} m/s")
    print()

    # ── Determine test targets ──
    if args.target:
        x, y = map(float, args.target.split(","))
        targets = [np.array([x, y, pos0[2]])]
    else:
        # Default: test in 4 directions + a multi-waypoint path
        targets = [
            # Single-waypoint tests in 4 cardinal directions (20m each)
            np.array([pos0[0] + 20, pos0[1], pos0[2]]),        # forward/east
            np.array([pos0[0] - 20, pos0[1], pos0[2]]),        # backward/west
            np.array([pos0[0], pos0[1] + 20, pos0[2]]),        # right/south
            np.array([pos0[0], pos0[1] - 20, pos0[2]]),        # left/north
        ]

    print(f"{'='*60}")
    print(f"  Testing {len(targets)} target(s)")
    print(f"{'='*60}\n")

    overall_ok = True
    for ti, target in enumerate(targets):
        print(f"--- Target {ti+1}: ({target[0]:.1f}, {target[1]:.1f}) ---")

        pos_before, _, _ = get_car_state(start_car)
        dist_before = float(np.linalg.norm(pos_before[:2] - target[:2]))
        print(f"  Distance before: {dist_before:.1f}m")

        # Single waypoint path
        follow_path(start_car, [target])

        pos_after, _, _ = get_car_state(start_car)
        dist_after = float(np.linalg.norm(pos_after[:2] - target[:2]))
        moved = float(np.linalg.norm(pos_after[:2] - pos_before[:2]))
        print(f"  Distance after:  {dist_after:.1f}m  (moved {moved:.1f}m)")
        print(f"  Position after:  ({pos_after[0]:.1f}, {pos_after[1]:.1f})")

        if dist_after < dist_before:
            print(f"  ✓ Got closer to target\n")
        else:
            print(f"  ✗ Did NOT get closer to target\n")
            overall_ok = False

    # ── Summary ──
    pos_final, _, _ = get_car_state(start_car)
    total_moved = float(np.linalg.norm(pos_final[:2] - pos0[:2]))
    print(f"{'='*60}")
    print(f"  Start:  ({pos0[0]:.1f}, {pos0[1]:.1f})")
    print(f"  Final:  ({pos_final[0]:.1f}, {pos_final[1]:.1f})")
    print(f"  Moved:  {total_moved:.1f}m total")
    print(f"  Result: {'PASS' if overall_ok else 'FAIL — check steering'}")
    print(f"{'='*60}")

    return overall_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
