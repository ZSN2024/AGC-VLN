#!/usr/bin/env python3
"""Air-ground collaborative closed-loop — UAV (modified SPF) + UGV (VLM path planning).

Flow:
  1. UAV runs modified SPF: downward camera → VLM locate truck → fly toward it
  2. UGV sends its CARLA coordinates → UAV annotates birdview with CAR + GOAL positions
  3. UAV sends annotated birdview back to UGV
  4. UGV uses VLM on the birdview to plan a driving route → executes drive actions
  5. Both agents operate in parallel; either reaching the goal (≤10m) = success

Usage:
  python collab_UAV_UGV/scripts/run_collab.py --episode-id town10hd_001
  python collab_UAV_UGV/scripts/run_collab.py --episode-id town10hd_001 --auto
"""

import base64, json, math, os, sys, threading, time
from dataclasses import dataclass
from pathlib import Path

import airsim, carla, cv2, httpx
import numpy as np
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

DRONE_ALTITUDE = 60.0
DOWNWARD_FOV = 108.0
SUCCESS_DIST = 10.0
DECISION_INTERVAL = 3.0

# ═══════════════════════════════════════════════════════════════════
# Shared state (thread-safe communication between UAV and UGV)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SharedState:
    lock: threading.Lock
    # Positions
    drone_pos: np.ndarray       # NED [x, y, z]
    drone_yaw: float
    ugv_pos: np.ndarray         # CARLA [x, y, z]
    goal_ned: np.ndarray        # NED [x, y, z]
    goal_carla: np.ndarray      # CARLA [x, y, z]
    # Birdview image — UAV annotates & shares, UGV reads
    annotated_image: np.ndarray | None
    image_timestamp: float
    # Drone state AT annotation time (for correct pixel→world conversion)
    annot_drone_pos: np.ndarray | None   # NED at annotation moment
    annot_drone_yaw: float
    # CAR/GOAL pixel coords in birdview (for VLM prompt)
    car_px: float; car_py: float
    goal_px: float; goal_py: float
    img_w: int; img_h: int
    # Control flags
    stop_event: threading.Event
    uav_success: bool
    ugv_success: bool
    uav_success_time: float | None
    ugv_success_time: float | None
    # Metrics
    uav_dc: int
    ugv_dc: int
    uav_path: list
    ugv_path: list


# ═══════════════════════════════════════════════════════════════════
# Config & episode loading
# ═══════════════════════════════════════════════════════════════════

def _project_root():
    return Path(__file__).resolve().parent.parent.parent


def load_config():
    cfg = json.loads(
        (_project_root() / "config.json").read_text()
    )["models"]["spf"]
    model = cfg["model"]
    route = "dashscope" if model.lower().startswith("qwen") else "ccode"
    env_var = "SPF_DASHSCOPE_API_KEY" if route == "dashscope" else "SPF_CCODE_API_KEY"
    cfg["route"] = route
    cfg["base_url"] = cfg["base_urls"][route]
    cfg["api_key"] = os.environ.get(env_var, "") or cfg["api_keys"].get(route, "")
    return cfg


def load_episode(ep_id):
    episodes_dir = _project_root() / "episodes"
    for name in ("_annotated.json", "_templates.json"):
        path = episodes_dir / f"town10hd{name}"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for ep in data["episodes"]:
                if ep["id"] == ep_id:
                    return ep
    raise FileNotFoundError(f"Episode {ep_id} not found")


# ═══════════════════════════════════════════════════════════════════
# Connection
# ═══════════════════════════════════════════════════════════════════

def connect():
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(10.0)
    world = cl.get_world()
    drone_actor = next(
        (a for a in world.get_actors() if "drone" in a.type_id.lower()), None
    )
    if drone_actor is None:
        raise RuntimeError("CarlaAir drone not found — is CarlaAir.sh running?")
    air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=30)
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


def carla_to_ned(loc, offset):
    return tuple(offset + np.array([loc.x, loc.y, -loc.z]))


# ═══════════════════════════════════════════════════════════════════
# Vehicle helpers
# ═══════════════════════════════════════════════════════════════════

def find_cars(world):
    """Return (start_car=Mini, goal_car=HGV)."""
    start = goal = None
    for v in world.get_actors().filter("vehicle.*"):
        if "carlacola" in v.type_id or "hgv" in v.type_id.lower():
            goal = v
        elif "mini" in v.type_id.lower():
            start = v
    return start, goal


def get_car_state(car):
    loc = car.get_location()
    vel = car.get_velocity()
    yaw = math.radians(car.get_transform().rotation.yaw)
    return np.array([loc.x, loc.y, loc.z]), yaw, math.sqrt(vel.x ** 2 + vel.y ** 2)


def spawn_car_camera(world, car):
    """FPV camera mounted on car hood. Returns (camera, latest_frame_list)."""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "540")
    bp.set_attribute("fov", "110")
    # ── color tuning: warmer white balance, lock auto-exposure ──
    bp.set_attribute("temp", "5500")              # default 6500K; lower=warmer (toward orange)
    bp.set_attribute("tint", "0")                 # neutral green↔magenta balance
    bp.set_attribute("exposure_mode", "manual")   # disable auto-exposure to match viewport
    bp.set_attribute("gamma", "2.2")
    tf = carla.Transform(carla.Location(x=1.8, z=1.3))
    cam = world.spawn_actor(bp, tf, attach_to=car)
    result = [None]

    def cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        result[0] = arr.reshape((img.height, img.width, 4))[:, :, :3]  # BGRA→BGR

    cam.listen(cb)
    time.sleep(0.3)
    return cam, result


def spawn_drone_camera(world, drone):
    """Third-person chase camera following the drone (simulation-environment view)."""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "540")
    bp.set_attribute("fov", "110")
    # Close behind-and-above the drone, tilted down — drone clearly visible,
    # and the ground target/car stay in frame once the drone descends to it.
    tf = carla.Transform(carla.Location(x=-8.0, z=4.0), carla.Rotation(pitch=-25.0))
    cam = world.spawn_actor(bp, tf, attach_to=drone)
    result = [None]

    def cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        result[0] = arr.reshape((img.height, img.width, 4))[:, :, :3]  # BGRA→BGR

    cam.listen(cb)
    time.sleep(0.3)
    return cam, result


# ═══════════════════════════════════════════════════════════════════
# UAV helpers
# ═══════════════════════════════════════════════════════════════════

def capture_downward(air):
    resp = air.simGetImages(
        [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
    )
    if not resp or resp[0].width == 0:
        return None
    return np.frombuffer(
        resp[0].image_data_uint8, dtype=np.uint8
    ).reshape(resp[0].height, resp[0].width, 3).copy()


def get_drone_state(air):
    s = air.getMultirotorState().kinematics_estimated
    return np.array([s.position.x_val, s.position.y_val, s.position.z_val]), \
        airsim.to_eularian_angles(s.orientation)[2]


def wait_stable(air, timeout=5.0):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        s = air.getMultirotorState().kinematics_estimated
        v = s.linear_velocity
        spd = math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2)
        pitch, roll, _ = airsim.to_eularian_angles(s.orientation)
        if spd < 0.3 and abs(math.degrees(pitch)) < 2 and abs(math.degrees(roll)) < 2:
            return
        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════
# Coordinate projection (NED ↔ image pixels)
# ═══════════════════════════════════════════════════════════════════

def pix_to_world(pt, drone_pos, drone_yaw, img_w, img_h, fov=DOWNWARD_FOV):
    """Pixel in image → world NED coordinates."""
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


def world_to_pix(world_xy, drone_pos, drone_yaw, img_w, img_h, fov=DOWNWARD_FOV):
    """World NED xy → pixel (x, y) in image, clipped to bounds."""
    h = abs(drone_pos[2])
    if h < 1.0:
        return img_w // 2, img_h // 2
    half_w = h * math.tan(math.radians(fov / 2))
    wx_off = world_xy[0] - drone_pos[0]
    wy_off = world_xy[1] - drone_pos[1]
    cy, sy = math.cos(drone_yaw), math.sin(drone_yaw)
    ned_x = cy * wx_off + sy * wy_off
    ned_y = -sy * wx_off + cy * wy_off
    dx = ned_y / half_w if half_w > 0 else 0.0
    dy = -ned_x / half_w if half_w > 0 else 0.0
    px = int((dx / 2 + 0.5) * img_w)
    py = int((dy / 2 + 0.5) * img_h)
    return max(0, min(img_w - 1, px)), max(0, min(img_h - 1, py))


# ═══════════════════════════════════════════════════════════════════
# Flight control
# ═══════════════════════════════════════════════════════════════════

def fly_horizontal(air, target, drone_pos, Kp=1.0, max_speed=5.0):
    delta = target[:2] - drone_pos[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 0.3:
        return
    speed = min(max_speed, Kp * dist)
    dur = max(1.0, dist / speed)
    vx, vy = delta[0] / dur, delta[1] / dur
    print(f"         fly: dist={dist:.0f}m speed={speed:.1f}m/s dur={dur:.1f}s")
    air.moveByVelocityAsync(
        float(vx), float(vy), 0.0, dur,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, 0),
    ).join()


def do_height(air, height_cmd, goal_ned_z):
    dp, _ = get_drone_state(air)
    above = max(0.0, goal_ned_z - dp[2])
    floor = 5.0
    if height_cmd == -1 and above > floor:
        print(f"         descending 10m (above_truck={above:.0f}m)")
        air.moveByVelocityAsync(0, 0, 5.0, 2.0).join()  # NED: +Z = down
    elif height_cmd == 1 and abs(dp[2]) < 80:
        print(f"         ascending 10m")
        air.moveByVelocityAsync(0, 0, -5.0, 2.0).join()  # NED: -Z = up
    else:
        print(f"         height hold")
    air.hoverAsync()
    wait_stable(air)


# ═══════════════════════════════════════════════════════════════════
# VLM calls
# ═══════════════════════════════════════════════════════════════════

def vlm_uav(client, cfg, img, goal_img_path):
    """UAV modified SPF: locate the orange truck in downward view."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    current_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    goal_b64 = None
    if goal_img_path and Path(goal_img_path).exists():
        goal_img = cv2.imread(str(goal_img_path))
        if goal_img is not None:
            goal_img = cv2.resize(goal_img, (640, 480))
            _, buf2 = cv2.imencode(".jpg", goal_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            goal_b64 = base64.b64encode(buf2.tobytes()).decode("ascii")

    prompt = (
        "Find the orange truck. Return ONLY valid JSON (no markdown, no explanation).\n"
        "Format: {\"point\": [x, y], \"height\": -1/0/1}\n"
        "x,y: pixel coordinates (0-1280, 0-960). height:-1=descend, 0=hold, 1=ascend.\n"
        "Always use height=-1 when truck is visible."
    )
    content = [{"type": "text", "text": prompt}]
    if goal_b64:
        content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + goal_b64}})
    content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + current_b64}})

    resp = client.chat.completions.create(
        model=cfg["model"], temperature=0.1, max_tokens=256,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.choices[0].message.content.strip()
    print(f"  UAV VLM: {text[:150]}")
    # Robust JSON extraction
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    # Try to find JSON object in case model wrapped it in text
    if "{" in text:
        text = text[text.index("{"):text.rindex("}") + 1]
    return json.loads(text)


def vlm_ugv(client, cfg, annotated_img, goal_img_path,
            car_px, car_py, goal_px, goal_py, img_w, img_h):
    """UGV path planning: VLM outputs 10-waypoint path in native pixel coords.
    Returns (result_dict, feedback_img)."""
    _, buf = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    goal_b64 = None
    if goal_img_path and Path(goal_img_path).exists():
        goal_img = cv2.imread(str(goal_img_path))
        if goal_img is not None:
            goal_img = cv2.resize(goal_img, (640, 480))
            _, buf2 = cv2.imencode(".jpg", goal_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            goal_b64 = base64.b64encode(buf2.tobytes()).decode("ascii")

    prompt = (
        "You are navigating a ground vehicle. The attached image is a BIRD'S-EYE VIEW "
        "from a drone overhead.\n\n"
        "On the image:\n"
        "- BLUE circle + 'CAR' label = your vehicle\n"
        "- RED circle + 'GOAL' label = the target truck you must reach\n"
        "- GREEN crosshair = drone position\n"
        "- Yellow line = direct line-of-sight (NOT drivable, for reference only)\n\n"
        f"IMAGE SIZE: {img_w} x {img_h} pixels.\n"
        f"CAR pixel position: ({car_px:.0f}, {car_py:.0f})\n"
        f"GOAL pixel position: ({goal_px:.0f}, {goal_py:.0f})\n\n"
        "TASK: Plan a drivable path along VISIBLE ROADS from CAR to GOAL.\n"
        "Generate exactly 10 waypoints in pixel coordinates. "
        "1st waypoint MUST equal the CAR position above. "
        "10th waypoint MUST equal the GOAL position above. "
        "Points 2-9 should be evenly spaced along a road-following route "
        "(follow road curves, turn at intersections, do NOT cut across buildings or off-road).\n\n"
        "Return ONLY valid JSON (no markdown, no explanation):\n"
        '{"path": [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5],'
        ' [x6,y6], [x7,y7], [x8,y8], [x9,y9], [x10,y10]]}'
    )
    content = [{"type": "text", "text": prompt}]
    if goal_b64:
        content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + goal_b64}})
    content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}})

    resp = client.chat.completions.create(
        model=cfg["model"], temperature=0.1, max_tokens=512,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.choices[0].message.content.strip()
    print(f"  UGV VLM: {text[:150]}")
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    if "{" in text:
        text = text[text.index("{"):text.rindex("}") + 1]
    result = json.loads(text)

    # Draw path on feedback image
    path = result.get("path", [])
    feedback_img = annotated_img.copy()
    for i in range(len(path) - 1):
        p1 = (int(path[i][0]), int(path[i][1]))
        p2 = (int(path[i + 1][0]), int(path[i + 1][1]))
        cv2.line(feedback_img, p1, p2, (0, 255, 0), 2)
    for i, pt in enumerate(path):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(feedback_img, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(feedback_img, str(i + 1), (px + 8, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return result, feedback_img


# ═══════════════════════════════════════════════════════════════════
# UGV path-following — 6 driving primitives (closed-loop)
# ═══════════════════════════════════════════════════════════════════

def _classify_action(angle_err):
    """Angle error → (action_name, steer, reverse)."""
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


def follow_path(car, world_waypoints, goal_xy=None):
    """Closed-loop: read position → adjust throttle/steer → stop when ≤2m from waypoint.
    If goal_xy is set, checks after each waypoint whether goal is reached (≤10m)."""
    STEP = 0.1
    STALL_THRESHOLD = 2.0   # seconds of no movement before triggering unstuck

    for idx, wp in enumerate(world_waypoints):
        wp_xy = wp[:2]
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

            if dist < 2.0:
                print(f"         wp{idx+1}/{len(world_waypoints)} ✓  dist={dist:.1f}m")
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.2)
                break

            wall_elapsed = time.monotonic() - t_start_wall
            sim_elapsed = car.get_world().get_snapshot().timestamp.elapsed_seconds - t_start_sim
            if sim_elapsed > 15.0:
                ratio = sim_elapsed / wall_elapsed if wall_elapsed > 0 else 1.0
                print(f"         wp{idx+1}/{len(world_waypoints)} ⚠ timeout  dist={dist:.1f}m  "
                      f"wall={wall_elapsed:.1f}s sim={sim_elapsed:.1f}s ratio={ratio:.2f}x")
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
                # Back up *while steering* so the nose swings toward the waypoint
                # (reverse steering is opposite to forward: steer away from the
                # target to bring the nose toward it). Straight reverse keeps the
                # wrong heading and just re-stalls.
                target_yaw = math.atan2(dy, dx)
                angle_err = math.degrees((target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
                rev_steer = -0.6 if angle_err > 0 else 0.6
                print(f"         ⚡ STUCK {stall_t:.1f}s — reverse-turn to unstick (err={angle_err:+.0f}deg)")
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.3)
                car.apply_control(carla.VehicleControl(throttle=0.8, steer=rev_steer, reverse=True))
                time.sleep(2.0)
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.3)
                stall_t = 0.0
                last_pos = None
                continue

            target_yaw = math.atan2(dy, dx)
            angle_err = math.degrees((target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
            action, steer, reverse = _classify_action(angle_err)

            throttle = min(1.0, max(0.3, dist / 10.0))
            if reverse:
                throttle = min(1.0, max(0.3, dist / 10.0))

            ctrl = carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse)
            car.apply_control(ctrl)

            if action != last_action:
                print(f"         wp{idx+1}/{len(world_waypoints)}: "
                      f"dist={dist:.0f}m err={angle_err:+.0f}deg "
                      f"-> {action} thr={throttle:.2f} steer={steer:+.1f}")
                last_action = action

            time.sleep(STEP)

        # Check goal after each waypoint
        if goal_xy is not None:
            pos_after, _, _ = get_car_state(car)
            goal_dist = float(np.linalg.norm(pos_after[:2] - goal_xy))
            if goal_dist < 10.0:
                print(f"\n  🏁 GOAL REACHED! dist={goal_dist:.1f}m  (at wp{idx+1}/{len(world_waypoints)})\n")
                car.apply_control(carla.VehicleControl(brake=1.0))
                return True

    car.apply_control(carla.VehicleControl(brake=1.0))
    time.sleep(0.3)
    return False


# ═══════════════════════════════════════════════════════════════════
# UAV thread — modified SPF + annotate birdview for UGV
# ═══════════════════════════════════════════════════════════════════

def uav_loop(state: SharedState, air, client, cfg, ep, offset, log_dir, log_f, auto_mode, sim_cam=None, sim_frame=None):
    """
    UAV runs modified SPF:
      1. Capture downward image
      2. Annotate birdview: mark CAR (blue) and GOAL (red) positions,
         draw direction line, overlay distance labels
      3. Share annotated image to UGV via shared state
      4. VLM locate truck → compute waypoint → fly → adjust height
    """
    goal = ep["goal"]
    goal_ned = np.array(carla_to_ned(
        carla.Location(x=goal["x"], y=goal["y"], z=goal["z"]), offset
    ))
    goal_photo = (
        Path(__file__).resolve().parent.parent / ep.get("goal_photo", "")
        if ep.get("goal_photo") else None
    )

    print(f"[UAV] started, goal_ned=({goal_ned[0]:.0f},{goal_ned[1]:.0f})")

    uav_done = False  # local flag: UAV reached goal, but keeps annotating for UGV
    next_decision = 0.0
    dc = 0

    try:
        while not state.stop_event.is_set():
            now = time.monotonic()
            drone_pos, drone_yaw = get_drone_state(air)
            d_xy = float(np.linalg.norm(drone_pos[:2] - goal_ned[:2]))
            d_z = abs(drone_pos[2] - goal_ned[2])
            dist = math.sqrt(d_xy ** 2 + d_z ** 2)

            # Update shared position
            with state.lock:
                state.drone_pos = drone_pos.copy()
                state.drone_yaw = drone_yaw

            # Success check — set flag, keep annotating for UGV
            if dist < SUCCESS_DIST and not uav_done:
                uav_done = True
                with state.lock:
                    state.uav_success = True
                    state.uav_success_time = time.monotonic()
                print(f"\n  [UAV] SUCCESS! dist={dist:.1f}m dc={dc}")
                if state.ugv_success:
                    state.stop_event.set()
                    break
                # Continue annotating — don't break

            if now >= next_decision:
                if not uav_done:
                    dc += 1
                    state.uav_dc = dc

                # Capture downward image
                img = capture_downward(air)
                if img is None:
                    time.sleep(0.1)
                    continue
                wait_stable(air)
                img = capture_downward(air)
                if img is None:
                    time.sleep(0.1)
                    continue

                h_img, w_img = img.shape[:2]

                # ── Step 2-3: Annotate birdview with UGV + goal positions ──
                annotated = img.copy()
                with state.lock:
                    ugv_carla = state.ugv_pos.copy()

                # Convert UGV CARLA position to NED
                ugv_ned = offset + np.array([ugv_carla[0], ugv_carla[1], -ugv_carla[2]])
                goal_ned_xy = goal_ned[:2]

                # Project UGV and goal positions to image pixels
                ugv_px, ugv_py = world_to_pix(ugv_ned[:2], drone_pos, drone_yaw, w_img, h_img)
                goal_px, goal_py = world_to_pix(goal_ned_xy, drone_pos, drone_yaw, w_img, h_img)

                # Draw annotations on birdview
                # Green crosshair = drone center
                cv2.line(annotated, (w_img // 2 - 30, h_img // 2), (w_img // 2 + 30, h_img // 2),
                         (0, 255, 0), 2)
                cv2.line(annotated, (w_img // 2, h_img // 2 - 30), (w_img // 2, h_img // 2 + 30),
                         (0, 255, 0), 2)
                # Blue circle = CAR (UGV)
                cv2.circle(annotated, (ugv_px, ugv_py), 25, (255, 150, 0), 3)
                cv2.circle(annotated, (ugv_px, ugv_py), 5, (255, 150, 0), -1)
                cv2.putText(annotated, "CAR", (ugv_px + 15, ugv_py - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
                # Red circle = GOAL
                cv2.circle(annotated, (goal_px, goal_py), 25, (0, 0, 255), 3)
                cv2.circle(annotated, (goal_px, goal_py), 5, (0, 0, 255), -1)
                cv2.putText(annotated, "GOAL", (goal_px + 15, goal_py - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Yellow line = direction from CAR to GOAL
                cv2.line(annotated, (ugv_px, ugv_py), (goal_px, goal_py), (0, 255, 255), 2)
                # Distance labels
                car_to_goal = float(np.linalg.norm(ugv_ned[:2] - goal_ned_xy))
                cv2.putText(annotated, f"UAV->goal:{d_xy:.0f}m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f"CAR->goal:{car_to_goal:.0f}m",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                # ── Share annotated image + pixel coords to UGV ──
                with state.lock:
                    state.annotated_image = annotated.copy()
                    state.image_timestamp = time.monotonic()
                    # CRITICAL: store drone state at annotation time for correct pixel→world
                    state.annot_drone_pos = drone_pos.copy()
                    state.annot_drone_yaw = drone_yaw
                    state.car_px = float(ugv_px); state.car_py = float(ugv_py)
                    state.goal_px = float(goal_px); state.goal_py = float(goal_py)
                    state.img_w = w_img; state.img_h = h_img

                # Save for logging
                if auto_mode:
                    cv2.imwrite(str(log_dir / f"step{dc:03d}_annotated.jpg"),
                                annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    # Drone-flying view (terminal-1 simulation environment)
                    if sim_frame is not None and sim_frame[0] is not None:
                        cv2.imwrite(str(log_dir / f"step{dc:03d}_sim.jpg"),
                                    sim_frame[0], [cv2.IMWRITE_JPEG_QUALITY, 85])

                # ── VLM + fly (skip if UAV already at goal) ──
                if not uav_done:
                    try:
                        if auto_mode:
                            cv2.imwrite(str(log_dir / f"step{dc:03d}_input.jpg"),
                                        img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        result = vlm_uav(client, cfg, img, goal_photo)
                        if auto_mode:
                            (log_dir / f"step{dc:03d}_uav_vlm.json").write_text(
                                json.dumps(result, indent=2), encoding="utf-8")
                            log_f.write(f"[UAV {dc}] dist={dist:.1f}m VLM: {json.dumps(result)}\n")
                    except Exception as e:
                        print(f"  [UAV {dc}] VLM err: {e}")
                        time.sleep(1.0)
                        next_decision = now + DECISION_INTERVAL
                        continue

                    pt = result["point"]
                    height_cmd = result.get("height", 0)
                    # VLM returns actual pixel coords (0-1280, 0-960) — no scaling needed
                    target = pix_to_world(pt, drone_pos, drone_yaw, w_img, h_img)

                    print(f"  [UAV {dc}] dist={dist:.0f}m pt=({pt[0]:.0f},{pt[1]:.0f}) height={height_cmd}")

                    fly_horizontal(air, target, drone_pos)
                    air.hoverAsync()
                    wait_stable(air)
                    do_height(air, height_cmd, goal_ned[2])

                    # Track path
                    np_pos, _ = get_drone_state(air)
                    with state.lock:
                        state.uav_path.append((time.monotonic(), np_pos.copy()))
                else:
                    # UAV done — just hover and track
                    air.hoverAsync()

                next_decision = now + DECISION_INTERVAL
            time.sleep(0.1)

    except Exception as e:
        print(f"  [UAV] error: {e}")
        state.stop_event.set()
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# UGV thread — send position → receive birdview → VLM → drive
# ═══════════════════════════════════════════════════════════════════

def ugv_loop(state: SharedState, start_car, world, client, cfg, ep,
             log_dir, log_f, auto_mode):
    """
    UGV collaboration loop:
      1. Send current CARLA position to shared state
      2. Wait for annotated birdview from UAV
      3. VLM path planning from birdview image
      4. Execute driving action (forward/turn/reverse/stop)
    """
    goal = ep["goal"]
    goal_photo = (
        Path(__file__).resolve().parent.parent / ep.get("goal_photo", "")
        if ep.get("goal_photo") else None
    )
    goal_carla = np.array([goal["x"], goal["y"], goal["z"]])

    print(f"[UGV] started, goal_carla=({goal_carla[0]:.0f},{goal_carla[1]:.0f})")

    # Spawn FPV camera on car
    car_cam, latest_fpv = spawn_car_camera(world, start_car)
    ugv_done = False  # local flag: UGV reached goal, but keeps updating position for UAV
    next_decision = 0.0
    dc = 0
    last_image_ts = 0.0

    try:
        while not state.stop_event.is_set():
            now = time.monotonic()
            pos, yaw, spd = get_car_state(start_car)
            dist = float(np.linalg.norm(pos[:2] - goal_carla[:2]))

            # ── Step 1: Send UGV position to UAV (via shared state) ──
            with state.lock:
                state.ugv_pos = pos.copy()
                state.goal_carla = goal_carla.copy()

            # Success check — set flag, keep updating position for UAV
            if dist < SUCCESS_DIST and not ugv_done:
                ugv_done = True
                with state.lock:
                    state.ugv_success = True
                    state.ugv_success_time = time.monotonic()
                print(f"\n  [UGV] SUCCESS! dist={dist:.1f}m dc={dc}")
                if state.uav_success:
                    state.stop_event.set()
                    break
                # Continue sending position — don't break

            if now >= next_decision:
                if not ugv_done:
                    dc += 1
                    state.ugv_dc = dc

                # ── Step 2-4: VLM + drive (skip if UGV already at goal) ──
                if not ugv_done:
                    # Wait for annotated birdview from UAV
                    annotated = None
                    wait_start = time.monotonic()
                    while time.monotonic() - wait_start < 10.0:
                        with state.lock:
                            if (state.annotated_image is not None
                                    and state.image_timestamp > last_image_ts):
                                annotated = state.annotated_image.copy()
                                last_image_ts = state.image_timestamp
                                break
                        if state.stop_event.is_set():
                            break
                        time.sleep(0.2)

                    if annotated is None:
                        print(f"  [UGV {dc}] no birdview yet, creeping forward...")
                        start_car.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
                        time.sleep(2.0)
                        start_car.apply_control(carla.VehicleControl(brake=1.0))
                        next_decision = now + 2.0
                        continue

                    # Save FPV for reference
                    if auto_mode and latest_fpv[0] is not None:
                        cv2.imwrite(str(log_dir / f"step{dc:03d}_fpv.jpg"),
                                    latest_fpv[0], [cv2.IMWRITE_JPEG_QUALITY, 85])

                    # VLM path planning
                    try:
                        if auto_mode:
                            cv2.imwrite(str(log_dir / f"step{dc:03d}_birdview.jpg"),
                                        annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        with state.lock:
                            cpx = state.car_px; cpy = state.car_py
                            gpx = state.goal_px; gpy = state.goal_py
                            iw = state.img_w; ih = state.img_h
                        result, feedback_img = vlm_ugv(client, cfg, annotated, goal_photo,
                                                        cpx, cpy, gpx, gpy, iw, ih)
                        cv2.imwrite(str(log_dir / f"step{dc:03d}_feedback.jpg"),
                                    feedback_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        (log_dir / f"step{dc:03d}_ugv_vlm.json").write_text(
                            json.dumps(result, indent=2), encoding="utf-8")
                        log_f.write(f"[UGV {dc}] dist={dist:.1f}m VLM: {json.dumps(result)}\n")
                    except Exception as e:
                        print(f"  [UGV {dc}] VLM err: {e}")
                        time.sleep(1.0)
                        next_decision = now + DECISION_INTERVAL
                        continue

                    # Pixel path → CARLA coordinates → execute
                    path = result.get("path", [])
                    # Sanity check: reject hallucinated paths
                    ok_path = len(path) >= 2
                    if ok_path:
                        p0, plast = path[0], path[-1]
                        aw, ah = annotated.shape[1], annotated.shape[0]
                        # Reject if first point is near (0,0) or way outside image
                        if (p0[0] < 10 or p0[1] < 10) or \
                           p0[0] < -aw or p0[0] > aw * 2 or \
                           p0[1] < -ah or p0[1] > ah * 2:
                            print(f"  [UGV {dc}] BAD PATH: first pt=({p0[0]:.0f},{p0[1]:.0f}) hallucinated, retry")
                            ok_path = False
                        # Verify first wp ≈ CAR position, last wp ≈ GOAL position
                        if ok_path:
                            with state.lock:
                                _cpx, _cpy = state.car_px, state.car_py
                                _gpx, _gpy = state.goal_px, state.goal_py
                            d_start = np.linalg.norm([p0[0] - _cpx, p0[1] - _cpy])
                            d_end = np.linalg.norm([plast[0] - _gpx, plast[1] - _gpy])
                            if d_start > 200 or d_end > 200:
                                print(f"  [UGV {dc}] BAD PATH: start_err={d_start:.0f}px end_err={d_end:.0f}px, retry")
                                ok_path = False
                    if not ok_path:
                        next_decision = now + 1.0
                        continue

                    with state.lock:
                        # Use drone state AT annotation time (not current) for pixel→world
                        dp = state.annot_drone_pos.copy() if state.annot_drone_pos is not None else state.drone_pos.copy()
                        dyaw = state.annot_drone_yaw
                        gn = state.goal_ned.copy()
                        gc = state.goal_carla.copy()

                    img_h, img_w = annotated.shape[:2]
                    ned_to_carla_xy = gn[:2] - gc[:2]
                    world_waypoints = []
                    for pt in path:
                        wp_ned = pix_to_world(pt, dp, dyaw, img_w, img_h)
                        wp_carla = np.array([wp_ned[0] - ned_to_carla_xy[0],
                                             wp_ned[1] - ned_to_carla_xy[1],
                                             wp_ned[2]])
                        world_waypoints.append(wp_carla)

                    print(f"  [UGV {dc}] executing {len(world_waypoints)} waypoints...")

                    pos_before = get_car_state(start_car)[0].copy()
                    goal_reached = follow_path(start_car, world_waypoints, goal_xy=gc[:2])
                    pos_after = get_car_state(start_car)[0]

                    # Early goal success via follow_path
                    if goal_reached:
                        ugv_done = True
                        with state.lock:
                            state.ugv_success = True
                            state.ugv_success_time = time.monotonic()
                        print(f"\n  [UGV] SUCCESS! dist={float(np.linalg.norm(pos_after[:2] - gc[:2])):.1f}m dc={dc}")
                        if state.uav_success:
                            state.stop_event.set()
                            break
                    else:
                        # Stall detection
                        moved = float(np.linalg.norm(pos_after[:2] - pos_before[:2]))
                        if moved < 1.0:
                            print(f"  [UGV {dc}] STALLED (moved {moved:.1f}m), auto-reverse 3m")
                            start_car.apply_control(carla.VehicleControl(throttle=0.5, reverse=True))
                            time.sleep(2.0)
                            start_car.apply_control(carla.VehicleControl(brake=1.0))

                    with state.lock:
                        state.ugv_path.append((time.monotonic(), pos_after.copy()))

                next_decision = now + DECISION_INTERVAL
            time.sleep(0.1)

    except Exception as e:
        print(f"  [UGV] error: {e}")
        state.stop_event.set()
        import traceback
        traceback.print_exc()
    finally:
        if car_cam:
            try:
                car_cam.stop()
                car_cam.destroy()
            except Exception:
                pass
        start_car.apply_control(carla.VehicleControl(brake=1.0))


# ═══════════════════════════════════════════════════════════════════
# Metrics & result saving
# ═══════════════════════════════════════════════════════════════════

def save_result(ok, elapsed, uav_dc, ugv_dc, uav_path, ugv_path,
                final_dist_uav, final_dist_ugv,
                uav_success_time, ugv_success_time, start_time,
                log_dir, log_f):
    def _path_len(path):
        m = 0.0
        for i in range(1, len(path)):
            m += float(np.linalg.norm(path[i][1][:2] - path[i - 1][1][:2]))
        return round(m, 1)

    # First-arrival time (either member): the "Time" metric uses the earlier success
    arrivals = [t for t in (uav_success_time, ugv_success_time) if t is not None]
    first_arrival = min(arrivals) if arrivals else None

    r = {
        "success": ok,
        "time_s": round(elapsed, 1),
        "uav_success": uav_success_time is not None,
        "ugv_success": ugv_success_time is not None,
        "uav_success_time_s": round(uav_success_time - start_time, 1) if uav_success_time else None,
        "ugv_success_time_s": round(ugv_success_time - start_time, 1) if ugv_success_time else None,
        "success_time_s": round(first_arrival - start_time, 1) if first_arrival else None,
        "uav_path_m": _path_len(uav_path),
        "ugv_path_m": _path_len(ugv_path),
        "uav_final_dist_m": round(final_dist_uav, 1),
        "ugv_final_dist_m": round(final_dist_ugv, 1),
        "uav_vlm_calls": uav_dc,
        "ugv_vlm_calls": ugv_dc,
        "total_vlm_calls": uav_dc + ugv_dc,
    }
    (log_dir / "result.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    log_f.write(f"\nresult: {json.dumps(r)}\n")
    log_f.close()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="Air-ground collaborative navigation")
    p.add_argument("--episode-id", required=True, help="Episode ID, e.g. town10hd_001")
    p.add_argument("--time-limit", type=float, default=180.0, help="Max time per episode (s)")
    p.add_argument("--auto", action="store_true", help="Auto mode: no interactive prompts")
    p.add_argument("--run-dir", default=None, help="Parent dir for this run's output (default: runs/)")
    args = p.parse_args()

    ep = load_episode(args.episode_id)
    cfg = load_config()

    # Setup logging
    runs_dir = Path(args.run_dir) if args.run_dir else (Path(__file__).resolve().parent.parent / "runs")
    log_dir = runs_dir / f"collab_{ep['id']}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = (log_dir / "log.txt").open("w", encoding="utf-8", buffering=1)
    log_f.write(f"Collab UAV+UGV — {ep['id']}  "
                f"goal=({ep['goal']['x']},{ep['goal']['y']})  "
                f"time_limit={args.time_limit}s\n\n")
    print(f"Logging to {log_dir}")

    # Connect to simulation
    cl, world, air, offset = connect()
    client = OpenAI(
        api_key=cfg["api_key"], base_url=cfg["base_url"],
        http_client=httpx.Client(proxy=None, trust_env=False), timeout=30.0,
    )

    goal = ep["goal"]
    goal_ned = np.array(carla_to_ned(
        carla.Location(x=goal["x"], y=goal["y"], z=goal["z"]), offset
    ))
    goal_carla = np.array([goal["x"], goal["y"], goal["z"]])

    # Find vehicles (Mini Cooper = UGV, HGV = goal truck)
    start_car, goal_car = find_cars(world)
    if start_car is None or goal_car is None:
        print("ERROR: vehicles not found — run setup_scene.py first")
        return False

    # Third-person camera following the drone (terminal-1 simulation view)
    drone_actor = next(
        (a for a in world.get_actors() if "drone" in a.type_id.lower()), None
    )
    sim_cam, sim_frame = (spawn_drone_camera(world, drone_actor)
                          if drone_actor is not None else (None, None))

    # Get initial positions
    drone_pos, drone_yaw = get_drone_state(air)
    ugv_pos, _, _ = get_car_state(start_car)

    # Initialize shared state
    state = SharedState(
        lock=threading.Lock(),
        drone_pos=drone_pos.copy(), drone_yaw=drone_yaw,
        ugv_pos=ugv_pos.copy(),
        goal_ned=goal_ned.copy(), goal_carla=goal_carla.copy(),
        annotated_image=None, image_timestamp=0.0,
        annot_drone_pos=None, annot_drone_yaw=0.0,
        car_px=0.0, car_py=0.0, goal_px=0.0, goal_py=0.0,
        img_w=0, img_h=0,
        stop_event=threading.Event(),
        uav_success=False, ugv_success=False,
        uav_success_time=None, ugv_success_time=None,
        uav_dc=0, ugv_dc=0,
        uav_path=[(time.monotonic(), drone_pos.copy())],
        ugv_path=[(time.monotonic(), ugv_pos.copy())],
    )

    print(f"\n{'=' * 55}")
    print(f"  Collaborative Nav — {ep['id']}  |  {ep['distance_m']:.0f}m")
    print(f"  Goal:  ({goal['x']:.0f}, {goal['y']:.0f})")
    print(f"  UAV:   Modified SPF search + annotate birdview")
    print(f"  UGV:   Send position → birdview → VLM → drive")
    print(f"{'=' * 55}\n")

    # Start UAV and UGV threads
    uav_th = threading.Thread(
        target=uav_loop,
        args=(state, air, client, cfg, ep, offset, log_dir, log_f, args.auto, sim_cam, sim_frame),
        daemon=True, name="UAV",
    )
    ugv_th = threading.Thread(
        target=ugv_loop,
        args=(state, start_car, world, client, cfg, ep, log_dir, log_f, args.auto),
        daemon=True, name="UGV",
    )

    started = time.monotonic()
    uav_th.start()
    ugv_th.start()

    # Wait for completion or timeout (polling, so Ctrl+C works)
    deadline = time.monotonic() + args.time_limit
    try:
        while time.monotonic() < deadline:
            if state.stop_event.is_set():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    finally:
        state.stop_event.set()

    elapsed = time.monotonic() - started

    # Final state (AirSim may be in disconnected state, don't crash)
    try:
        drone_pos_f, _ = get_drone_state(air)
        dist_uav = float(np.linalg.norm(drone_pos_f - goal_ned))
    except Exception:
        # Fall back to last known position from shared state
        dist_uav = float(np.linalg.norm(state.drone_pos[:2] - goal_ned[:2]))
    try:
        ugv_pos_f, _, _ = get_car_state(start_car)
        dist_ugv = float(np.linalg.norm(ugv_pos_f[:2] - goal_carla[:2]))
    except Exception:
        # Use last known UGV position from shared state
        dist_ugv = float(np.linalg.norm(state.ugv_pos[:2] - goal_carla[:2]))
    ok = state.uav_success or state.ugv_success

    print(f"\n{'=' * 55}")
    print(f"  {'SUCCESS' if ok else 'FAILED'}  time={elapsed:.0f}s")
    print(f"  UAV: success={state.uav_success}  "
          f"time={'%.1fs'%((state.uav_success_time-started)) if state.uav_success_time else '--'}  "
          f"dist={dist_uav:.1f}m  dc={state.uav_dc}")
    print(f"  UGV: success={state.ugv_success}  "
          f"time={'%.1fs'%((state.ugv_success_time-started)) if state.ugv_success_time else '--'}  "
          f"dist={dist_ugv:.1f}m  dc={state.ugv_dc}")
    print(f"{'=' * 55}")

    save_result(ok, elapsed, state.uav_dc, state.ugv_dc,
                state.uav_path, state.ugv_path,
                dist_uav, dist_ugv,
                state.uav_success_time, state.ugv_success_time,
                started, log_dir, log_f)

    if sim_cam is not None:
        try:
            sim_cam.stop()
            sim_cam.destroy()
        except Exception:
            pass
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
