#!/usr/bin/env python3
"""UGV 单独调试 — 无人机悬停标注鸟瞰图，汽车根据标注图 VLM 规划驾驶。

Usage:
  python collab_UAV_UGV/scripts/debug_ugv.py --episode-id town10hd_001
  python collab_UAV_UGV/scripts/debug_ugv.py --episode-id town10hd_001 --auto
"""

import base64, json, math, os, sys, threading, time
from dataclasses import dataclass
from pathlib import Path

import airsim, carla, cv2, httpx
import numpy as np
from openai import OpenAI

DOWNWARD_FOV = 108.0
SUCCESS_DIST = 10.0
DECISION_INTERVAL = 3.0
ANNOTATE_INTERVAL = 1.0  # 标注更新频率（比决策间隔快）


@dataclass
class SharedState:
    lock: threading.Lock
    drone_pos: np.ndarray
    drone_yaw: float
    ugv_pos: np.ndarray
    goal_ned: np.ndarray
    goal_carla: np.ndarray
    annotated_image: np.ndarray | None
    image_timestamp: float
    # CAR/GOAL pixel coords in the latest birdview (for VLM prompt)
    car_px: float
    car_py: float
    goal_px: float
    goal_py: float
    # Image dimensions
    img_w: int
    img_h: int
    stop_event: threading.Event
    ugv_success: bool
    ugv_dc: int
    ugv_path: list


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
    episodes_dir = Path(__file__).resolve().parent.parent / "episodes"
    for name in ("_annotated.json", "_templates.json"):
        path = episodes_dir / f"town10hd{name}"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for ep in data["episodes"]:
                if ep["id"] == ep_id:
                    return ep
    raise FileNotFoundError(f"Episode {ep_id} not found")


def connect():
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(10.0)
    world = cl.get_world()
    drone_actor = next((a for a in world.get_actors() if "drone" in a.type_id.lower()), None)
    if drone_actor is None:
        raise RuntimeError("CarlaAir drone not found")
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


def find_cars(world):
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


def capture_downward(air):
    resp = air.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    if not resp or resp[0].width == 0:
        return None
    return np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8).reshape(
        resp[0].height, resp[0].width, 3).copy()


def get_drone_state(air):
    s = air.getMultirotorState().kinematics_estimated
    return np.array([s.position.x_val, s.position.y_val, s.position.z_val]), \
        airsim.to_eularian_angles(s.orientation)[2]


def world_to_pix(world_xy, drone_pos, drone_yaw, img_w, img_h, fov=DOWNWARD_FOV):
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


# ═══════════════════════════════════════════════════════════════
# VLM
# ═══════════════════════════════════════════════════════════════

def vlm_ugv(client, cfg, annotated_img, goal_img_path, dc, dist,
            car_px, car_py, goal_px, goal_py, img_w, img_h):
    """VLM path planning: birdview + explicit CAR/GOAL pixel coords → 10-waypoint path.

    Pixel coordinates use the image's native resolution (img_w × img_h).
    """
    _, buf = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    goal_b64 = None
    if goal_img_path and Path(goal_img_path).exists():
        goal_img = cv2.imread(str(goal_img_path))
        if goal_img is not None:
            goal_img = cv2.resize(goal_img, (640, 480))
            _, buf2 = cv2.imencode(".jpg", goal_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            goal_b64 = base64.b64encode(buf2.tobytes()).decode("ascii")

    h, w = annotated_img.shape[:2]
    display_img = annotated_img.copy()
    cv2.putText(display_img, f"UGV #{dc} | dist={dist:.0f}m | Sending to VLM...",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imshow("1. Input to VLM (birdview)", cv2.resize(display_img, (960, 720)))
    cv2.waitKey(800)

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
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]

    # ── 解析路径 ──
    result = json.loads(text)
    path = result.get("path", [])
    n_pts = len(path)

    # ── 打印 ──
    print(f"\n  {'─' * 40}")
    print(f"  [UGV #{dc}] dist={dist:.0f}m  VLM 路径 ({n_pts} pts):")
    for i, pt in enumerate(path):
        print(f"    {i+1:>2}. ({pt[0]:.0f}, {pt[1]:.0f})")
    print(f"  {'─' * 40}\n")

    # ── 在图上绘制路径 (VLM returns native pixel coords) ──
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

    cv2.rectangle(feedback_img, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.putText(feedback_img, f"Path: {n_pts} waypoints along roads", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(feedback_img, "[ENTER]=执行  [S]=跳过  [Q]=退出", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return result, feedback_img


# ═══════════════════════════════════════════════════════════════
# 像素 → 世界坐标（和 run_collab.py 一致）
# ═══════════════════════════════════════════════════════════════

def pix_to_world(pt, drone_pos, drone_yaw, img_w, img_h, fov=DOWNWARD_FOV):
    """Pixel (in image coords) → world NED."""
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


# ═══════════════════════════════════════════════════════════════
# UGV 路径跟随 — 6 个驾驶原语（闭环控制）
# ═══════════════════════════════════════════════════════════════

def _classify_action(angle_err):
    """角度误差 → (动作名, 方向盘, 是否倒车)"""
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


def follow_path(car, world_waypoints, verbose=True, goal_xy=None):
    """闭环控制：持续读位置 → 调整油门/方向 → 到达目标点（≤2m）才停。
    如果 goal_xy 不为 None，每到达一个 waypoint 检查是否已在目标 10m 内。"""
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
                if verbose:
                    print(f"         wp{idx+1}/{len(world_waypoints)} ✓  dist={dist:.1f}m")
                car.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.2)
                break

            sim_elapsed = car.get_world().get_snapshot().timestamp.elapsed_seconds - t_start_sim
            if sim_elapsed > 15.0:
                wall_elapsed = time.monotonic() - t_start_wall
                ratio = sim_elapsed / wall_elapsed if wall_elapsed > 0 else 1.0
                if verbose:
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
                if verbose:
                    print(f"         ⚡ STUCK {stall_t:.1f}s — reversing 5m to unstick")
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

            target_yaw = math.atan2(dy, dx)
            angle_err = math.degrees((target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
            action, steer, reverse = _classify_action(angle_err)

            throttle = min(1.0, max(0.3, dist / 10.0))
            if reverse:
                throttle = min(1.0, max(0.3, dist / 10.0))

            ctrl = carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse)
            car.apply_control(ctrl)

            if action != last_action:
                if verbose:
                    print(f"         wp{idx+1}/{len(world_waypoints)}: "
                          f"dist={dist:.0f}m err={angle_err:+.0f}deg "
                          f"-> {action} thr={throttle:.2f} steer={steer:+.1f}")
                last_action = action

            time.sleep(STEP)

        # 每个 waypoint 到达后检查是否已到目标
        if goal_xy is not None:
            pos_after, _, _ = get_car_state(car)
            goal_dist = float(np.linalg.norm(pos_after[:2] - goal_xy))
            if goal_dist < 10.0:
                if verbose:
                    print(f"\n  🏁 GOAL REACHED! dist={goal_dist:.1f}m  (at wp{idx+1}/{len(world_waypoints)})\n")
                car.apply_control(carla.VehicleControl(brake=1.0))
                return True  # 提前结束

    car.apply_control(carla.VehicleControl(brake=1.0))
    time.sleep(0.3)
    return False  # 路径走完但未到目标


# ═══════════════════════════════════════════════════════════════
# UAV 悬停线程 — 只捕获下视图 + 标注，不飞行
# ═══════════════════════════════════════════════════════════════

def uav_hover_loop(state: SharedState, air, offset, log_dir, auto_mode):
    """UAV 悬停在原点：持续捕获下视 + 标注 CAR/GOAL 位置发给 UGV。"""
    print("[UAV] hover mode — 悬停标注，不飞行")
    annotate_count = 0

    try:
        while not state.stop_event.is_set():
            # 保持悬停
            air.hoverAsync()

            drone_pos, drone_yaw = get_drone_state(air)
            with state.lock:
                state.drone_pos = drone_pos.copy()
                state.drone_yaw = drone_yaw

            # 捕获下视图
            img = capture_downward(air)
            if img is None:
                time.sleep(0.1)
                continue

            h_img, w_img = img.shape[:2]

            # 读 UGV 位置
            with state.lock:
                ugv_carla = state.ugv_pos.copy()
                goal_ned = state.goal_ned.copy()

            # 坐标转换
            ugv_ned = offset + np.array([ugv_carla[0], ugv_carla[1], -ugv_carla[2]])
            goal_ned_xy = goal_ned[:2]

            # 投影到像素
            ugv_px, ugv_py = world_to_pix(ugv_ned[:2], drone_pos, drone_yaw, w_img, h_img)
            goal_px, goal_py = world_to_pix(goal_ned_xy, drone_pos, drone_yaw, w_img, h_img)

            # 标注
            annotated = img.copy()
            # 绿色十字 = 无人机位置
            cv2.line(annotated, (w_img // 2 - 30, h_img // 2), (w_img // 2 + 30, h_img // 2),
                     (0, 255, 0), 2)
            cv2.line(annotated, (w_img // 2, h_img // 2 - 30), (w_img // 2, h_img // 2 + 30),
                     (0, 255, 0), 2)
            # 蓝圈 = CAR
            cv2.circle(annotated, (ugv_px, ugv_py), 25, (255, 150, 0), 3)
            cv2.circle(annotated, (ugv_px, ugv_py), 5, (255, 150, 0), -1)
            cv2.putText(annotated, "CAR", (ugv_px + 15, ugv_py - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
            # 红圈 = GOAL
            cv2.circle(annotated, (goal_px, goal_py), 25, (0, 0, 255), 3)
            cv2.circle(annotated, (goal_px, goal_py), 5, (0, 0, 255), -1)
            cv2.putText(annotated, "GOAL", (goal_px + 15, goal_py - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 黄线 = CAR→GOAL 方向
            cv2.line(annotated, (ugv_px, ugv_py), (goal_px, goal_py), (0, 255, 255), 2)
            # 距离标签
            car_to_goal = float(np.linalg.norm(ugv_ned[:2] - goal_ned_xy))
            cv2.putText(annotated, f"UAV HOVER (debug)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(annotated, f"CAR->goal:{car_to_goal:.0f}m", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            # 共享给 UGV（图片 + CAR/GOAL 像素坐标）
            with state.lock:
                state.annotated_image = annotated.copy()
                state.image_timestamp = time.monotonic()
                state.car_px = float(ugv_px)
                state.car_py = float(ugv_py)
                state.goal_px = float(goal_px)
                state.goal_py = float(goal_py)
                state.img_w = w_img
                state.img_h = h_img

            annotate_count += 1
            if annotate_count % 10 == 0:
                print(f"  [UAV hover] annotated #{annotate_count}  CAR->goal={car_to_goal:.0f}m")

            if auto_mode and annotate_count % 5 == 0:
                cv2.imwrite(str(log_dir / f"hover_{annotate_count:04d}.jpg"),
                            annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

            time.sleep(ANNOTATE_INTERVAL)

    except Exception as e:
        print(f"  [UAV hover] error: {e}")
        state.stop_event.set()
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# UGV 线程
# ═══════════════════════════════════════════════════════════════

def ugv_loop(state: SharedState, start_car, world, client, cfg, ep,
             log_dir, log_f, auto_mode):
    goal = ep["goal"]
    goal_photo = (
        Path(__file__).resolve().parent.parent / ep.get("goal_photo", "")
        if ep.get("goal_photo") else None
    )
    goal_carla = np.array([goal["x"], goal["y"], goal["z"]])

    print(f"[UGV] started, goal=({goal_carla[0]:.0f},{goal_carla[1]:.0f})")

    car_cam, latest_fpv = spawn_car_camera(world, start_car)
    next_decision = 0.0
    dc = 0
    last_image_ts = 0.0

    try:
        while not state.stop_event.is_set():
            now = time.monotonic()
            pos, yaw, spd = get_car_state(start_car)
            dist = float(np.linalg.norm(pos[:2] - goal_carla[:2]))

            # 发送位置给 UAV
            with state.lock:
                state.ugv_pos = pos.copy()
                state.goal_carla = goal_carla.copy()

            # 成功判定
            if dist < SUCCESS_DIST:
                with state.lock:
                    state.ugv_success = True
                state.stop_event.set()
                print(f"\n  [UGV] SUCCESS! dist={dist:.1f}m dc={dc}")
                break

            if now >= next_decision:
                dc += 1
                state.ugv_dc = dc

                # 等 UAV 标注图
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
                    print(f"  [UGV {dc}] no birdview, creeping...")
                    start_car.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
                    time.sleep(2.0)
                    start_car.apply_control(carla.VehicleControl(brake=1.0))
                    next_decision = now + 2.0
                    continue

                # 保存 VLM 输入（标注鸟瞰图）
                cv2.imwrite(str(log_dir / f"step{dc:03d}_birdview.jpg"),
                            annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                # 保存 FPV
                if latest_fpv[0] is not None:
                    cv2.imwrite(str(log_dir / f"step{dc:03d}_fpv.jpg"),
                                latest_fpv[0], [cv2.IMWRITE_JPEG_QUALITY, 85])

                # VLM 路径规划
                try:
                    with state.lock:
                        cpx = state.car_px; cpy = state.car_py
                        gpx = state.goal_px; gpy = state.goal_py
                        iw = state.img_w; ih = state.img_h
                    result, feedback_img = vlm_ugv(
                        client, cfg, annotated, goal_photo, dc, dist,
                        cpx, cpy, gpx, gpy, iw, ih)

                    # 保存 VLM 输出（画了路径的反馈图）
                    cv2.imwrite(str(log_dir / f"step{dc:03d}_feedback.jpg"),
                                feedback_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

                    if auto_mode:
                        # 自动模式：显示 1 秒不等待
                        cv2.imshow("2. VLM feedback (auto)", cv2.resize(feedback_img, (960, 720)))
                        cv2.waitKey(1000)
                    else:
                        # 交互模式：等待用户确认
                        cv2.imshow("2. VLM feedback - CONFIRM?", cv2.resize(feedback_img, (960, 720)))
                        print(f"  [UGV #{dc}] 等待确认: ENTER=执行  S=跳过  Q=退出")
                        key = 0
                        while key not in (13, ord('s'), ord('q'), 27):
                            key = cv2.waitKey(100) & 0xFF
                        cv2.destroyWindow("2. VLM feedback - CONFIRM?")

                        if key == ord('q') or key == 27:
                            print(f"  [UGV #{dc}] 用户退出")
                            state.stop_event.set()
                            break
                        if key == ord('s'):
                            print(f"  [UGV #{dc}] 用户跳过, 继续下一轮")
                            next_decision = now + 1.0
                            continue
                        # ENTER → 执行

                    # 保存 VLM 原始响应
                    (log_dir / f"step{dc:03d}_vlm.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8")
                    log_f.write(f"[UGV {dc}] dist={dist:.1f}m VLM: {json.dumps(result)}\n")
                except Exception as e:
                    print(f"  [UGV {dc}] VLM err: {e}")
                    time.sleep(1.0)
                    next_decision = now + DECISION_INTERVAL
                    continue

                # ── 像素路径 → 世界坐标 → 执行 ──
                path = result.get("path", [])
                if len(path) < 2:
                    print(f"  [UGV {dc}] path too short ({len(path)} pts), skip")
                    next_decision = now + 1.0
                    continue

                # 读取无人机状态做坐标转换
                with state.lock:
                    dp = state.drone_pos.copy()
                    dyaw = state.drone_yaw
                    gn = state.goal_ned.copy()
                    gc = state.goal_carla.copy()

                img_h, img_w = annotated.shape[:2]
                # pix_to_world returns NED coords, but follow_path needs CARLA coords.
                # Convert: CARLA_xy = NED_xy - offset_xy
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

                # 提前到达目标 → 直接结束
                if goal_reached:
                    dist_after = float(np.linalg.norm(pos_after[:2] - gc[:2]))
                    if dist_after < SUCCESS_DIST:
                        with state.lock:
                            state.ugv_success = True
                        state.stop_event.set()
                        print(f"\n  [UGV] SUCCESS! dist={dist_after:.1f}m dc={dc}")
                        break

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


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="UGV debug — UAV hover + annotate, UGV VLM drive")
    p.add_argument("--episode-id", required=True)
    p.add_argument("--time-limit", type=float, default=180.0)
    p.add_argument("--auto", action="store_true")
    args = p.parse_args()

    ep = load_episode(args.episode_id)
    cfg = load_config()

    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    log_dir = runs_dir / f"debug_ugv_{ep['id']}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = (log_dir / "log.txt").open("w", encoding="utf-8")
    log_f.write(f"Debug UGV — UAV hover — {ep['id']}  "
                f"goal=({ep['goal']['x']},{ep['goal']['y']})  "
                f"time_limit={args.time_limit}s\n\n")
    print(f"Logging to {log_dir}")

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

    start_car, goal_car = find_cars(world)
    if start_car is None or goal_car is None:
        print("ERROR: vehicles not found — run setup_scene.py first")
        return False

    drone_pos, drone_yaw = get_drone_state(air)
    ugv_pos, _, _ = get_car_state(start_car)

    state = SharedState(
        lock=threading.Lock(),
        drone_pos=drone_pos.copy(), drone_yaw=drone_yaw,
        ugv_pos=ugv_pos.copy(),
        goal_ned=goal_ned.copy(), goal_carla=goal_carla.copy(),
        annotated_image=None, image_timestamp=0.0,
        car_px=0.0, car_py=0.0, goal_px=0.0, goal_py=0.0,
        img_w=0, img_h=0,
        stop_event=threading.Event(),
        ugv_success=False,
        ugv_dc=0,
        ugv_path=[(time.monotonic(), ugv_pos.copy())],
    )

    print(f"\n{'=' * 55}")
    print(f"  UGV Debug — {ep['id']}  |  {ep['distance_m']:.0f}m")
    print(f"  Goal:  ({goal['x']:.0f}, {goal['y']:.0f})")
    print(f"  UAV:   HOVER only — 悬停标注鸟瞰图")
    print(f"  UGV:   Birdview → VLM → drive")
    print(f"{'=' * 55}\n")

    uav_th = threading.Thread(
        target=uav_hover_loop,
        args=(state, air, offset, log_dir, args.auto),
        daemon=True, name="UAV-hover",
    )
    ugv_th = threading.Thread(
        target=ugv_loop,
        args=(state, start_car, world, client, cfg, ep, log_dir, log_f, args.auto),
        daemon=True, name="UGV",
    )

    started = time.monotonic()
    uav_th.start()
    ugv_th.start()

    ugv_th.join(timeout=args.time_limit)
    state.stop_event.set()

    elapsed = time.monotonic() - started
    ugv_pos_f, _, _ = get_car_state(start_car)
    dist_ugv = float(np.linalg.norm(ugv_pos_f[:2] - goal_carla[:2]))
    ok = state.ugv_success

    print(f"\n{'=' * 55}")
    print(f"  {'SUCCESS' if ok else 'FAILED'}  time={elapsed:.0f}s")
    print(f"  UGV: success={state.ugv_success} dist={dist_ugv:.1f}m dc={state.ugv_dc}")
    print(f"{'=' * 55}")

    ugv_m = 0.0
    for i in range(1, len(state.ugv_path)):
        ugv_m += float(np.linalg.norm(state.ugv_path[i][1][:2] - state.ugv_path[i - 1][1][:2]))

    r = {
        "mode": "ugv_debug_uav_hover",
        "success": ok,
        "time_s": round(elapsed, 1),
        "ugv_path_m": round(ugv_m, 1),
        "ugv_final_dist_m": round(dist_ugv, 1),
        "ugv_vlm_calls": state.ugv_dc,
    }
    (log_dir / "result.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    log_f.write(f"\nresult: {json.dumps(r)}\n")
    log_f.close()
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
