#!/usr/bin/env python3
"""UAV + UGV 多视角实时查看 — 无人机前视/下视 + 汽车 FPV/Chase 四画面.

Usage:
    conda activate carlaAir
    python collab_UAV_UGV/scripts/show_dual_view.py              # 四画面 (默认)
    python collab_UAV_UGV/scripts/show_dual_view.py --view drone # 只看无人机
    python collab_UAV_UGV/scripts/show_dual_view.py --view car   # 只看汽车

键盘: Q/ESC=退出  V=切换视角  M=开关小地图  S=截图
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

import airsim
import carla
import cv2
import numpy as np
import pygame
import time

MINIMAP_SIZE = 240
MINIMAP_RANGE = 40.0
MARGIN = 10


# ═══════════════════════════════════════════════════════════════════
#  AirSim capture
# ═══════════════════════════════════════════════════════════════════

def _capture(air: airsim.MultirotorClient, camera: str) -> np.ndarray | None:
    responses = air.simGetImages([airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)])
    if not responses or responses[0].width == 0:
        return None
    raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
        responses[0].height, responses[0].width, 3
    )
    return raw[:, :, ::-1]  # AirSim Scene returns BGR; convert to RGB


# ═══════════════════════════════════════════════════════════════════
#  Minimap
# ═══════════════════════════════════════════════════════════════════

def _mm_xy(ned: np.ndarray, center: np.ndarray, s: float, cx: float, cy: float) -> tuple[int, int]:
    return (
        int(np.clip((ned[1] - center[1]) * s + cx, -MINIMAP_SIZE, MINIMAP_SIZE * 2)),
        int(np.clip((center[0] - ned[0]) * s + cy, -MINIMAP_SIZE, MINIMAP_SIZE * 2)),
    )


def _draw_triangle(surf: pygame.Surface, x: float, y: float, yaw: float, sz: float, color: tuple):
    pts = [(x + math.cos(yaw + math.pi * 0.8) * sz, y - math.sin(yaw + math.pi * 0.8) * sz),
           (x + math.cos(yaw) * sz * 1.6, y - math.sin(yaw) * sz * 1.6),
           (x + math.cos(yaw - math.pi * 0.8) * sz, y - math.sin(yaw - math.pi * 0.8) * sz)]
    pygame.draw.polygon(surf, color, pts)


def _minimap(drone_ned: np.ndarray, drone_yaw: float,
             truck_ned: np.ndarray | None, truck_yaw: float,
             cargo_ned: np.ndarray | None, font_sm) -> pygame.Surface:
    mm = pygame.Surface((MINIMAP_SIZE, MINIMAP_SIZE), pygame.SRCALPHA)
    mm.fill((10, 10, 16, 210))
    cx = cy = MINIMAP_SIZE / 2.0
    s = MINIMAP_SIZE / (2.0 * MINIMAP_RANGE)

    for m in range(-int(MINIMAP_RANGE), int(MINIMAP_RANGE) + 10, 10):
        px, _ = _mm_xy(np.array([0., float(m)]), drone_ned, s, cx, cy)
        pygame.draw.line(mm, (44, 44, 50, 130), (px, 0), (px, MINIMAP_SIZE), 1)
        _, py = _mm_xy(np.array([float(m), 0.]), drone_ned, s, cx, cy)
        pygame.draw.line(mm, (44, 44, 50, 130), (0, py), (MINIMAP_SIZE, py), 1)

    if truck_ned is not None:
        tx, ty = _mm_xy(truck_ned, drone_ned, s, cx, cy)
        ts = max(3, 6.0 * s)
        rect = pygame.Surface((ts, ts // 2), pygame.SRCALPHA)
        rect.fill((80, 160, 255, 200))
        rotated = pygame.transform.rotate(rect, -math.degrees(truck_yaw))
        mm.blit(rotated, (tx - rotated.get_width() // 2, ty - rotated.get_height() // 2))
        if cargo_ned is not None:
            cpx, cpy = _mm_xy(cargo_ned, drone_ned, s, cx, cy)
            pygame.draw.circle(mm, (255, 200, 0, 200), (cpx, cpy), 3)

    dx, dy = _mm_xy(drone_ned, drone_ned, s, cx, cy)
    _draw_triangle(mm, dx, dy, drone_yaw, 8.0, (0, 255, 100, 240))

    fov_len = 7.0 * s
    fov_half = math.radians(45)
    fov_pts = [(dx, dy)]
    for a in [drone_yaw - fov_half, drone_yaw - fov_half * 0.5, drone_yaw,
              drone_yaw + fov_half * 0.5, drone_yaw + fov_half]:
        fov_pts.append((dx + math.cos(a) * fov_len, dy - math.sin(a) * fov_len))
    if len(fov_pts) >= 3:
        pygame.draw.polygon(mm, (0, 255, 100, 50), fov_pts)

    # scale bar
    bar_m, bar_px = 10, int(10 * s)
    bar_x, bar_y = MINIMAP_SIZE - bar_px - 12, MINIMAP_SIZE - 15
    pygame.draw.line(mm, (180, 180, 180), (bar_x, bar_y), (bar_x + bar_px, bar_y), 3)
    mm.blit(font_sm.render(f"{bar_m}m", True, (200, 200, 200)), (bar_x + bar_px // 2 - 8, bar_y - 14))
    mm.blit(font_sm.render("N", True, (255, 255, 255)), (6, 4))

    if truck_ned is not None:
        d = float(np.linalg.norm(drone_ned[:2] - truck_ned[:2]))
        h = drone_ned[2] - truck_ned[2]
        mm.blit(font_sm.render(f"D→T:{d:.1f}m  H:{h:.1f}m", True, (170, 170, 170)),
                (6, MINIMAP_SIZE - 30))

    pygame.draw.rect(mm, (100, 100, 100), mm.get_rect(), 1)
    return mm


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Drone + Car dual-camera viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--view", choices=("all3", "drone", "car"), default="all3")
    args = parser.parse_args()

    air = airsim.MultirotorClient(ip=args.host, port=args.port, timeout_value=20)
    air.confirmConnection()
    print(f"AirSim: {args.host}:{args.port}")

    # Reset drone yaw to 0 (north) at startup
    try:
        state = air.getMultirotorState().kinematics_estimated
        pos = state.position
        orientation = airsim.to_quaternion(0.0, 0.0, 0.0)  # yaw=0=north
        air.simSetVehiclePose(airsim.Pose(pos, orientation), ignore_collision=True)
        time.sleep(0.2)
        air.moveByVelocityAsync(0, 0, 0, 0.5).join()
        print("Drone yaw reset to 0 (north)")
    except Exception as e:
        print(f"Drone reset: {e}")

    carla_world = None
    car_camera = None
    car_chase_cam = None
    latest_car_img = [None]
    latest_chase_img = [None]
    try:
        cc = carla.Client(args.host, args.carla_port)
        cc.set_timeout(5.0)
        carla_world = cc.get_world()
        print(f"CARLA : {args.host}:{args.carla_port}")
        print("CARLA connected. Car camera will attach when vehicle spawns.")
    except Exception as e:
        print(f"CARLA unavailable: {e}")

    def _spawn_car_cameras(world):
        """Attach FPV + chase cameras to the UGV (Mini Cooper), NOT the goal truck.

        Vehicle type IDs (from setup_scene.py):
          - START_TRUCK = "vehicle.mini.cooper_s"   → UGV (driving car)
          - GOAL_TRUCK  = "vehicle.carlamotors.carlacola" → HGV (target)
        """
        nonlocal car_camera, car_chase_cam
        vehicles = list(world.get_actors().filter("vehicle.*"))
        if len(vehicles) < 2:
            return
        # 找 Mini Cooper：排除 HGV/Carlacola
        ugv = None
        for v in vehicles:
            tid = v.type_id.lower()
            if "carlacola" in tid or "hgv" in tid:
                continue
            ugv = v
            break
        if ugv is None:
            return

        # FPV — driver hood view
        fpv_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        fpv_bp.set_attribute("image_size_x", "640"); fpv_bp.set_attribute("image_size_y", "480")
        fpv_bp.set_attribute("fov", "110")
        fpv_bp.set_attribute("temp", "5500")              # warmer white balance (default 6500K)
        fpv_bp.set_attribute("tint", "0")
        fpv_bp.set_attribute("exposure_mode", "manual")
        fpv_bp.set_attribute("gamma", "2.2")
        fpv_tf = carla.Transform(carla.Location(x=1.8, z=1.3), carla.Rotation(pitch=0))
        car_camera = world.spawn_actor(fpv_bp, fpv_tf, attach_to=ugv)
        def fpv_cb(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            latest_car_img[0] = arr.reshape((img.height, img.width, 4))[:,:,:3][:, :, ::-1]
        car_camera.listen(fpv_cb)

        # Chase — behind + above
        chase_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        chase_bp.set_attribute("image_size_x", "640"); chase_bp.set_attribute("image_size_y", "480")
        chase_bp.set_attribute("fov", "100")
        chase_bp.set_attribute("temp", "5500")            # warmer white balance (default 6500K)
        chase_bp.set_attribute("tint", "0")
        chase_bp.set_attribute("exposure_mode", "manual")
        chase_bp.set_attribute("gamma", "2.2")
        chase_tf = carla.Transform(carla.Location(x=-8.0, z=5.0), carla.Rotation(pitch=-15))
        car_chase_cam = world.spawn_actor(chase_bp, chase_tf, attach_to=ugv)
        def chase_cb(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            latest_chase_img[0] = arr.reshape((img.height, img.width, 4))[:,:,:3][:, :, ::-1]
        car_chase_cam.listen(chase_cb)

        print(f"Car cams: FPV + chase on {ugv.type_id} (UGV)")

    pygame.init()
    W, H = args.width, args.height
    display = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13, bold=True)
    font_sm = pygame.font.SysFont("monospace", 11, bold=False)

    running = True
    show_mm = False
    view = args.view

    fps_hist: list[float] = []
    last_t = time.monotonic()

    car_cam_retry = 0
    while running:
        # Retry camera attach: None, or dead sensor from previous scene
        if carla_world is not None and car_cam_retry % 60 == 0:
            try:
                if car_camera is not None:
                    car_camera.stop(); car_camera.destroy()
                if car_chase_cam is not None:
                    car_chase_cam.stop(); car_chase_cam.destroy()
            except Exception:
                pass
            car_camera = None; car_chase_cam = None
            latest_car_img[0] = None; latest_chase_img[0] = None
        if car_camera is None and carla_world is not None:
            try:
                _spawn_car_cameras(carla_world)
            except Exception:
                pass
        car_cam_retry += 1

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if ev.key == pygame.K_v:
                    view = {"all3": "drone", "drone": "car", "car": "all3"}[view]
                if ev.key == pygame.K_m:
                    show_mm = not show_mm
                if ev.key == pygame.K_s:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("screenshots", exist_ok=True)
                    pygame.image.save(display, f"screenshots/drone_{ts}.png")
                    print(f"  [screenshot] screenshots/drone_{ts}.png")

        # ── Capture both cameras ──
        front = None; down = None
        try:
            front = _capture(air, "0")
        except Exception:
            pass
        if view in ("all3", "drone"):
            try:
                down = _capture(air, "1")
            except Exception:
                pass

        # ── FPS ──
        now = time.monotonic()
        dt = now - last_t
        last_t = now
        fps_hist.append(1.0 / dt if dt > 0 else 0)
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        fps = sum(fps_hist) / len(fps_hist)

        # ── Drone state ──
        drone_ned = np.zeros(3)
        drone_yaw = 0.0
        try:
            s = air.getMultirotorState()
            p = s.kinematics_estimated.position
            drone_ned = np.array([p.x_val, p.y_val, p.z_val])
            _, _, drone_yaw = airsim.to_eularian_angles(s.kinematics_estimated.orientation)
        except Exception:
            pass

        # ── UGV (Mini Cooper) state for minimap ──
        ugv_ned, ugv_yaw, cargo_ned = None, 0.0, None
        if carla_world is not None:
            try:
                for actor in carla_world.get_actors().filter("vehicle.*"):
                    tid = actor.type_id.lower()
                    if "carlacola" in tid or "hgv" in tid:
                        continue  # 跳过 HGV 卡车
                    loc = actor.get_location()
                    rot = actor.get_transform().rotation
                    ugv_ned = np.array([loc.x, loc.y, -loc.z])
                    ugv_yaw = math.radians(rot.yaw)
                    fwd = np.array([math.cos(ugv_yaw), math.sin(ugv_yaw)])
                    cargo_ned = ugv_ned + np.array([fwd[0] * (-2.3), fwd[1] * (-2.3), -1.15])
                    break
            except Exception:
                pass

        # ── Layout ──
        display.fill((16, 16, 16))

        if view == "all3":
            pane_w = W // 4
            pane_h = H - 26
        else:
            pane_w = W // 2
            pane_h = H - 26

        def blit_img(img, x, y, w, h, label, label_color):
            if img is not None:
                resized = cv2.resize(img, (w, h))
                display.blit(pygame.surfarray.make_surface(resized.swapaxes(0, 1)), (x, y))
            else:
                pygame.draw.rect(display, (30, 30, 30), (x, y, w, h))
            display.blit(font_sm.render(label, True, label_color), (x + 6, y + 4))

        if view == "all3":
            # 4-panel: Drone Front | Drone Down | UGV FPV | UGV Chase
            blit_img(front, pane_w*0, 0, pane_w, pane_h, "Drone Front", (100, 255, 100))
            pygame.draw.line(display, (80, 80, 80), (pane_w, 0), (pane_w, pane_h), 2)
            blit_img(down, pane_w*1+1, 0, pane_w, pane_h, "Drone Down", (100, 200, 255))
            pygame.draw.line(display, (80, 80, 80), (pane_w*2, 0), (pane_w*2, pane_h), 2)
            blit_img(latest_car_img[0], pane_w*2+1, 0, pane_w, pane_h, "UGV FPV", (255, 200, 100))
            pygame.draw.line(display, (80, 80, 80), (pane_w*3, 0), (pane_w*3, pane_h), 2)
            blit_img(latest_chase_img[0], pane_w*3+1, 0, pane_w, pane_h, "UGV Chase", (200, 255, 150))
        elif view == "car":
            blit_img(latest_car_img[0], 0, 0, pane_w, pane_h, "UGV FPV", (255, 200, 100))
            blit_img(latest_chase_img[0], pane_w+1, 0, pane_w, pane_h, "UGV Chase", (200, 255, 150))
        elif view == "drone":
            blit_img(front, 0, 0, pane_w, pane_h, "Drone Front", (100, 255, 100))
            blit_img(down, pane_w+1, 0, pane_w, pane_h, "Drone Down", (100, 200, 255))

        # ── Minimap ──
        if show_mm:
            mm = _minimap(drone_ned, drone_yaw, ugv_ned, ugv_yaw, cargo_ned, font_sm)
            display.blit(mm, (W - MINIMAP_SIZE - MARGIN, MARGIN))

        # ── Status bar ──
        bar = pygame.Surface((W, 26))
        bar.set_alpha(200)
        bar.fill((18, 18, 22))
        display.blit(bar, (0, H - 26))
        ti = f"UGV:[{ugv_ned[0]:.1f},{ugv_ned[1]:.1f}]" if ugv_ned is not None else "UGV:--"
        status_text = (
            f"FPS:{fps:5.1f}  NED:[{drone_ned[0]:.1f},{drone_ned[1]:.1f},{drone_ned[2]:.1f}]"
            f"  Yaw:{math.degrees(drone_yaw):.0f}°  {ti}"
            f"  |  V:视角({view})  M:小地图  S:截图  Q:退出"
        )
        display.blit(font.render(status_text, True, (210, 210, 210)), (6, H - 22))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
