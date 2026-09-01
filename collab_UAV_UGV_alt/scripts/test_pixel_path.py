#!/usr/bin/env python3
"""离线测试：给定 10 个像素路径点 → 坐标转换打印 → 逐点闭环到位控制（每点确认）。

Usage:
  conda activate carlaAir
  python collab_UAV_UGV/scripts/test_pixel_path.py
  python collab_UAV_UGV/scripts/test_pixel_path.py --auto  # 跳过确认，自动执行
"""

import argparse, math, sys, time
import airsim, carla
import numpy as np

DOWNWARD_FOV = 108.0

# ═══════════════════════════════════════════════════════════════
#  10 个测试像素路径点（模拟 VLM 输出）
#  坐标系: 图片像素 (x, y)，原点在左上角
# ═══════════════════════════════════════════════════════════════
TEST_PIXEL_PATH = [
    [430, 600],   # 起点 ≈ CAR
    [420, 560],
    [400, 500],
    [380, 440],
    [360, 380],
    [340, 330],
    [310, 280],
    [280, 240],
    [250, 200],
    [220, 170],   # 终点 ≈ GOAL
]


# ═══════════════════════════════════════════════════════════════
#  Helpers (same logic as debug_ugv.py)
# ═══════════════════════════════════════════════════════════════

def connect():
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(10.0)
    world = cl.get_world()
    drone_actor = next((a for a in world.get_actors() if "drone" in a.type_id.lower()), None)
    if drone_actor is None:
        raise RuntimeError("CarlaAir drone not found")
    air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=15)
    air.confirmConnection()
    air.enableApiControl(True)
    air.armDisarm(True)
    try:
        if air.getMultirotorState().landed_state == airsim.LandedState.Landed:
            air.takeoffAsync().join()
        air.moveByVelocityAsync(0, 0, 0, 0.5).join()
    except Exception:
        pass
    ap = air.getMultirotorState().kinematics_estimated.position
    dl = drone_actor.get_location()
    offset = np.array([ap.x_val - dl.x, ap.y_val - dl.y, ap.z_val + dl.z])
    return cl, world, air, offset


def get_drone_state(air):
    s = air.getMultirotorState().kinematics_estimated
    return np.array([s.position.x_val, s.position.y_val, s.position.z_val]), \
        airsim.to_eularian_angles(s.orientation)[2]


def get_car_state(car):
    loc = car.get_location()
    vel = car.get_velocity()
    yaw = math.radians(car.get_transform().rotation.yaw)
    return np.array([loc.x, loc.y, loc.z]), yaw, math.sqrt(vel.x ** 2 + vel.y ** 2)


def pix_to_world(pt, drone_pos, drone_yaw, img_w, img_h, fov=DOWNWARD_FOV):
    """Pixel → world NED."""
    h = abs(drone_pos[2])
    dx = (pt[0] / img_w - 0.5) * 2
    dy = (pt[1] / img_h - 0.5) * 2
    half_w = h * math.tan(math.radians(fov / 2))
    ned_x = -dy * half_w
    ned_y = dx * half_w
    cy, sy = math.cos(drone_yaw), math.sin(drone_yaw)
    return np.array([
        drone_pos[0] + ned_x * cy - ned_y * sy,
        drone_pos[1] + ned_x * sy + ned_y * cy,
        drone_pos[2],
    ])


def _classify_action(angle_err):
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


def drive_to_point(car, target_xy, label=""):
    """闭环控制开到目标 CARLA 坐标点 (x, y)。"""
    STEP = 0.1
    t_start = time.monotonic()
    last_action = None

    while True:
        pos, yaw, spd = get_car_state(car)
        dx = target_xy[0] - pos[0]
        dy = target_xy[1] - pos[1]
        dist = float(np.linalg.norm([dx, dy]))

        if dist < 2.0:
            print(f"       ✓ 到达 {label}  dist={dist:.1f}m  speed={spd:.1f}m/s")
            car.apply_control(carla.VehicleControl(brake=1.0))
            time.sleep(0.2)
            return True

        if time.monotonic() - t_start > 20.0:
            print(f"       ⚠ 超时 {label}  dist={dist:.1f}m")
            car.apply_control(carla.VehicleControl(brake=1.0))
            time.sleep(0.2)
            return False

        target_yaw = math.atan2(dy, dx)
        angle_err = math.degrees((target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
        action, steer, reverse = _classify_action(angle_err)

        throttle = min(0.7, max(0.2, dist / 15.0))
        if reverse:
            throttle = min(0.5, max(0.2, dist / 20.0))

        ctrl = carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse)
        car.apply_control(ctrl)

        if action != last_action:
            print(f"       dist={dist:.1f}m  err={angle_err:+.0f}°  "
                  f"-> {action:>14s}  thr={throttle:.2f}  steer={steer:+.1f}")
            last_action = action

        time.sleep(STEP)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Offline test: pixel path → CARLA → drive point-by-point")
    parser.add_argument("--auto", action="store_true", help="Auto mode: no per-point confirmation")
    args = parser.parse_args()

    # ── Connect ──
    cl, world, air, offset = connect()
    print(f"[OK] CARLA + AirSim connected\n")

    # ── Find car ──
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
    print(f"Car: {start_car.type_id}")

    # ── Read drone state ──
    drone_pos, drone_yaw = get_drone_state(air)
    print(f"Drone NED:  ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})")
    print(f"Drone yaw:  {math.degrees(drone_yaw):.1f}°")

    # ── Read car state ──
    car_pos, car_yaw, _ = get_car_state(start_car)
    print(f"Car CARLA:  ({car_pos[0]:.1f}, {car_pos[1]:.1f})")
    print(f"Car yaw:    {math.degrees(car_yaw):.1f}°")

    # ── Image dimensions (capture one frame to get actual size) ──
    resp = air.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    if resp and resp[0].width > 0:
        img_h, img_w = resp[0].height, resp[0].width
    else:
        img_w, img_h = 640, 480  # fallback
    print(f"Downward camera: {img_w}x{img_h}")

    # ── Offset for NED→CARLA conversion ──
    # CARLA → NED: ned = offset + [carla_x, carla_y, -carla_z]
    # NED → CARLA: carla_xy = ned_xy - offset_xy
    offset_xy = offset[:2]

    print(f"\n{'='*65}")
    print(f"  Pixel path → NED → CARLA 转换")
    print(f"  Image:  {img_w}×{img_h}  FOV={DOWNWARD_FOV}°  offset_xy=({offset_xy[0]:.0f},{offset_xy[1]:.0f})")
    print(f"{'='*65}")

    # ── Step 1: Convert all points and print ──
    carla_waypoints = []
    for i, pt in enumerate(TEST_PIXEL_PATH):
        wp_ned = pix_to_world(pt, drone_pos, drone_yaw, img_w, img_h)
        wp_carla_xy = wp_ned[:2] - offset_xy
        carla_waypoints.append(wp_carla_xy)
        print(f"  pt{i+1:>2}: pixel ({pt[0]:>4}, {pt[1]:>4})  "
              f"→ NED ({wp_ned[0]:>7.1f}, {wp_ned[1]:>7.1f})  "
              f"→ CARLA ({wp_carla_xy[0]:>7.1f}, {wp_carla_xy[1]:>7.1f})")

    # Distance summary
    total_dist = 0.0
    for i in range(1, len(carla_waypoints)):
        total_dist += float(np.linalg.norm(carla_waypoints[i] - carla_waypoints[i-1]))
    car_to_first = float(np.linalg.norm(carla_waypoints[0] - car_pos[:2]))
    print(f"\n  Car → pt1:  {car_to_first:.1f}m")
    print(f"  Path total: {total_dist:.1f}m  ({len(carla_waypoints)} waypoints)")

    # ── Step 2: Drive point by point ──
    print(f"\n{'='*65}")
    print(f"  逐点到位控制  {'(AUTO)' if args.auto else '(ENTER=下一分 / Q=退出)'}")
    print(f"{'='*65}\n")

    for i, wp in enumerate(carla_waypoints):
        print(f"─── Waypoint {i+1}/{len(carla_waypoints)}: CARLA ({wp[0]:.1f}, {wp[1]:.1f}) ───")

        if not args.auto:
            key = input("      [ENTER]=执行  [Q]=退出: ").strip().lower()
            if key == 'q':
                print("      用户退出")
                break

        pos_before, _, _ = get_car_state(start_car)
        ok = drive_to_point(start_car, wp, label=f"pt{i+1}")
        pos_after, _, _ = get_car_state(start_car)
        moved = float(np.linalg.norm(pos_after[:2] - pos_before[:2]))
        remaining = float(np.linalg.norm(pos_after[:2] - wp))
        print(f"      移动了 {moved:.1f}m, 距目标还差 {remaining:.1f}m  {'✓' if ok else '✗'}\n")

    # ── Summary ──
    pos_final, _, _ = get_car_state(start_car)
    total_moved = float(np.linalg.norm(pos_final[:2] - car_pos[:2]))
    final_to_last = float(np.linalg.norm(pos_final[:2] - carla_waypoints[-1]))
    print(f"{'='*65}")
    print(f"  起点:  ({car_pos[0]:.1f}, {car_pos[1]:.1f})")
    print(f"  终点:  ({pos_final[0]:.1f}, {pos_final[1]:.1f})")
    print(f"  总移动: {total_moved:.1f}m")
    print(f"  距最终目标: {final_to_last:.1f}m")
    print(f"{'='*65}")

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
