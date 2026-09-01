#!/usr/bin/env python3
"""Scene setup for collab_UAV_UGV — spawn vehicles, position drone, keep scene alive.

Usage:
  python collab_UAV_UGV/scripts/setup_scene.py
  python collab_UAV_UGV/scripts/setup_scene.py --episode-id town10hd_001
"""

import argparse, json, math, sys, time, cv2
from pathlib import Path

import airsim, carla
import numpy as np

START_TRUCK = "vehicle.mini.cooper_s"
GOAL_TRUCK = "vehicle.carlamotors.carlacola"
DRONE_ALTITUDE = 60.0


def connect():
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(10.0)
    world = cl.get_world()

    drone = next((a for a in world.get_actors() if "drone" in a.type_id.lower()), None)
    if drone is None:
        raise RuntimeError("CarlaAir drone not found")

    air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=15)
    air.confirmConnection()
    # Re-sync AirSim + CARLA drone state back to its spawn point. Without this,
    # the AirSim kinematics estimate drifts from the CARLA actor transform over
    # repeated flights, so the recomputed offset below slowly goes wrong and the
    # UAV gets teleported to the wrong place after a few episodes.
    air.reset()
    air.enableApiControl(True)
    air.armDisarm(True)
    try:
        if air.getMultirotorState().landed_state == airsim.LandedState.Landed:
            air.takeoffAsync().join()
    except Exception:
        pass
    try:
        air.moveByVelocityAsync(0, 0, 0, 0.5).join()
    except Exception:
        pass

    ap = air.getMultirotorState().kinematics_estimated.position
    dl = drone.get_location()
    offset = np.array([ap.x_val - dl.x, ap.y_val - dl.y, ap.z_val + dl.z])
    return cl, world, air, offset


def carla_to_ned(loc, offset):
    ned = offset + np.array([loc.x, loc.y, -loc.z])
    return float(ned[0]), float(ned[1]), float(ned[2])


def clear_vehicles(world):
    for v in world.get_actors().filter("vehicle.*"):
        try:
            v.destroy()
        except RuntimeError:
            pass
    time.sleep(0.3)


def spawn_vehicle(world, bp_name, x, y, z, yaw=0.0):
    bp = world.get_blueprint_library().find(bp_name)
    tf = carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z) + 0.3),
        carla.Rotation(yaw=float(yaw)),
    )
    return world.try_spawn_actor(bp, tf)


def position_drone(air, offset, x, y, z, yaw=0.0):
    nx, ny, nz = carla_to_ned(carla.Location(x=x, y=y, z=z), offset)
    orientation = airsim.to_quaternion(0.0, 0.0, math.radians(yaw))
    air.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(nx, ny, nz), orientation),
        ignore_collision=True,
    )
    time.sleep(0.3)
    try:
        air.moveByVelocityAsync(0, 0, 0, 0.5).join()
    except Exception:
        pass
    time.sleep(0.1)


def pick_episode(episodes_dir: Path):
    files = sorted(episodes_dir.glob("*_templates.json")) + sorted(episodes_dir.glob("*_episodes.json"))
    files = list(dict.fromkeys(files))
    if not files:
        print("No episode files found in", episodes_dir)
        sys.exit(1)
    print("\nAvailable episode files:")
    for i, f in enumerate(files):
        print(f"  {i + 1:>2}. {f.name}")
    choice = input(f"\nSelect [1-{len(files)}, Enter=1]: ").strip()
    idx = int(choice) - 1 if choice else 0
    f = files[max(0, min(len(files) - 1, idx))]

    data = json.loads(f.read_text(encoding="utf-8"))
    episodes = data["episodes"]
    print(f"\n{len(episodes)} episodes:")
    for i, ep in enumerate(episodes):
        g = ep["goal"]
        us = ep["ugv_spawn"]
        print(f"  {i + 1:>2}. {ep['id']}  #{us['index']} -> ({g['x']:.0f},{g['y']:.0f})  {ep['distance_m']:.0f}m")
    choice = input(f"\nSelect [1-{len(episodes)}, Enter=1]: ").strip()
    idx = int(choice) - 1 if choice else 0
    return episodes[max(0, min(len(episodes) - 1, idx))]


def main():
    p = argparse.ArgumentParser(description="Scene setup for collab_UAV_UGV")
    p.add_argument("--episode-file", default=None, help="Path to episode JSON")
    p.add_argument("--episode-id", default=None, help="Episode ID to load")
    p.add_argument("--headless", action="store_true", help="Skip the overhead viewer window (no GUI)")
    p.add_argument("--altitude", type=float, default=DRONE_ALTITUDE, help="Drone altitude above UGV start (m)")
    args = p.parse_args()

    episodes_dir = Path(__file__).resolve().parent.parent / "episodes"

    if args.episode_file and args.episode_id:
        data = json.loads(Path(args.episode_file).read_text(encoding="utf-8"))
        ep = next(e for e in data["episodes"] if e["id"] == args.episode_id)
    else:
        ep = pick_episode(episodes_dir)

    cl, world, air, offset = connect()
    clear_vehicles(world)
    time.sleep(0.3)

    us = ep["ugv_spawn"]
    g = ep["goal"]

    # Spawn vehicles: Mini Cooper (UGV) + HGV (goal truck)
    start_car = spawn_vehicle(world, START_TRUCK, us["x"], us["y"], us["z"], us["yaw"])
    if start_car is None:
        print("ERROR: failed to spawn start car (Mini Cooper)")
        sys.exit(1)
    goal_car = spawn_vehicle(world, GOAL_TRUCK, g["x"], g["y"], g["z"], 0.0)
    if goal_car is None:
        print("ERROR: failed to spawn goal car (HGV)")
        sys.exit(1)

    # Position drone above start, facing north
    position_drone(air, offset, us["x"], us["y"], us["z"] + args.altitude, yaw=0.0)
    print(f"  Drone reset: facing north (yaw=0)")
    time.sleep(0.3)

    drone_state = air.getMultirotorState().kinematics_estimated.position
    print(f"\n{'=' * 55}")
    print(f"  Scene Ready — {ep['id']}  |  {ep['distance_m']:.0f}m")
    print(f"  UGV (Mini):  #{us['index']}  ({us['x']:.0f}, {us['y']:.0f})")
    print(f"  Goal (HGV):             ({g['x']:.0f}, {g['y']:.0f})")
    print(f"  Drone:       {args.altitude:.0f}m above start")
    print(f"{'=' * 55}")
    print(f"\n  Next: python collab_UAV_UGV/scripts/run_collab.py --episode-id {ep['id']} --auto")
    print(f"  Overhead window open. Press Ctrl+C to stop.\n")

    if args.headless:
        print("  Headless: viewer skipped, keeping scene alive.")
        return

    # ── Overhead camera (static, high above midpoint) ──
    mid_x = (us["x"] + g["x"]) / 2
    mid_y = (us["y"] + g["y"]) / 2
    alt = 150.0

    oh_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    oh_bp.set_attribute("image_size_x", "640")
    oh_bp.set_attribute("image_size_y", "640")
    oh_bp.set_attribute("fov", "110")
    oh_tf = carla.Transform(
        carla.Location(x=mid_x, y=mid_y, z=alt),
        carla.Rotation(pitch=-90),
    )
    overhead_cam = world.spawn_actor(oh_bp, oh_tf)
    latest_oh = [None]

    def oh_cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        latest_oh[0] = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]

    overhead_cam.listen(oh_cb)

    # Marker helpers
    fov, hw = 110.0, alt * math.tan(math.radians(110.0 / 2))
    scale = 320.0 / hw

    def proj(wx, wy):
        px = int(320 + (wx - mid_x) * scale)
        py = int(320 - (wy - mid_y) * scale)
        return max(0, min(639, px)), max(0, min(639, py))

    try:
        while True:
            if latest_oh[0] is not None:
                oh = latest_oh[0].copy()
                gx, gy = proj(g["x"], g["y"])
                sx, sy = proj(us["x"], us["y"])
                # START (Mini Cooper / UGV)
                cv2.circle(oh, (sx, sy), 12, (255, 150, 0), 2)
                cv2.circle(oh, (sx, sy), 3, (255, 150, 0), -1)
                cv2.putText(oh, "UGV", (sx + 10, sy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
                # GOAL (HGV truck)
                cv2.circle(oh, (gx, gy), 15, (0, 0, 255), 2)
                cv2.circle(oh, (gx, gy), 4, (0, 0, 255), -1)
                cv2.putText(oh, "GOAL", (gx + 10, gy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Direction line
                cv2.line(oh, (sx, sy), (gx, gy), (0, 255, 255), 1)
                cv2.putText(oh, f"{ep['id']} | {ep['distance_m']:.0f}m", (5, 630),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow(f"Overhead - {ep['id']}", oh)

            if cv2.waitKey(50) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if overhead_cam:
            try:
                overhead_cam.stop()
                overhead_cam.destroy()
            except Exception:
                pass
        print("\n  Cleaning up vehicles...")
        clear_vehicles(world)
        print("  Done.")


if __name__ == "__main__":
    main()
