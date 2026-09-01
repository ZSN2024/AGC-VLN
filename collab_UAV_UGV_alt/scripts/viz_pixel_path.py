#!/usr/bin/env python3
"""Generate 3 visualization images for pixel→NED→CARLA conversion check."""
import math, sys, time
import airsim, carla, cv2
import numpy as np

DOWNWARD_FOV = 108.0
TEST_PIXEL_PATH = [
    [430, 600], [420, 560], [400, 500], [380, 440], [360, 380],
    [340, 330], [310, 280], [280, 240], [250, 200], [220, 170],
]
OUT_DIR = "/home/zsn/VLN/carlaAir_experiments/collab_UAV_UGV/runs"

# ── Connect ──
cl = carla.Client("127.0.0.1", 2000); cl.set_timeout(10.0)
world = cl.get_world()
air = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=15)
air.confirmConnection()
print("[OK] Connected")

ap = air.getMultirotorState().kinematics_estimated.position
drone_actor = next(a for a in world.get_actors() if "drone" in a.type_id.lower())
dl = drone_actor.get_location()
offset = np.array([ap.x_val - dl.x, ap.y_val - dl.y, ap.z_val + dl.z])
offset_xy = offset[:2]

# ── Drone state ──
s = air.getMultirotorState().kinematics_estimated
drone_pos = np.array([s.position.x_val, s.position.y_val, s.position.z_val])
drone_yaw = airsim.to_eularian_angles(s.orientation)[2]

# ── Capture downward image ──
resp = air.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
img = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8).reshape(resp[0].height, resp[0].width, 3).copy()
h_img, w_img = img.shape[:2]

print(f"Drone NED: ({drone_pos[0]:.0f},{drone_pos[1]:.0f},{drone_pos[2]:.0f}) yaw={math.degrees(drone_yaw):.0f}°")
print(f"Image: {w_img}x{h_img}  offset_xy=({offset_xy[0]:.0f},{offset_xy[1]:.0f})")

# ── Convert pixel → NED → CARLA ──
def pix_to_world(pt, dp, dy, iw, ih):
    h_ = abs(dp[2])
    dx = (pt[0] / iw - 0.5) * 2; dy = (pt[1] / ih - 0.5) * 2
    half_w = h_ * math.tan(math.radians(DOWNWARD_FOV / 2))
    nx = -dy * half_w; ny = dx * half_w
    cy, sy = math.cos(dy), math.sin(dy)
    return np.array([dp[0] + nx * cy - ny * sy, dp[1] + nx * sy + ny * cy, dp[2]])

ned_pts = []; carla_pts = []
for pt in TEST_PIXEL_PATH:
    wp_n = pix_to_world(pt, drone_pos, drone_yaw, w_img, h_img)
    ned_pts.append(wp_n[:2])
    carla_pts.append(wp_n[:2] - offset_xy)

# ══════════════════════════════════════════════════════
# Image 1: Birdview + pixel path overlay
# ══════════════════════════════════════════════════════
img1 = img.copy()
for i in range(len(TEST_PIXEL_PATH) - 1):
    p1 = tuple(TEST_PIXEL_PATH[i]); p2 = tuple(TEST_PIXEL_PATH[i+1])
    cv2.line(img1, p1, p2, (0, 255, 0), 3)
for i, pt in enumerate(TEST_PIXEL_PATH):
    cv2.circle(img1, pt, 8, (0, 255, 0), -1)
    cv2.putText(img1, str(i+1), (pt[0]+10, pt[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# Center crosshair
cv2.line(img1, (w_img//2-30, h_img//2), (w_img//2+30, h_img//2), (0,255,0), 2)
cv2.line(img1, (w_img//2, h_img//2-30), (w_img//2, h_img//2+30), (0,255,0), 2)
cv2.putText(img1, "PIXEL PATH (raw coords)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite(f"{OUT_DIR}/test_pixel_path_1_pixel.png", img1)
print("Saved: test_pixel_path_1_pixel.png")

# ══════════════════════════════════════════════════════
# Image 2: NED + CARLA side-by-side world plot
# ══════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# NED
ned_arr = np.array(ned_pts)
ax1.plot(ned_arr[:, 0], ned_arr[:, 1], 'go-', markersize=8, linewidth=2)
for i, pt in enumerate(ned_pts):
    ax1.annotate(str(i+1), (pt[0], pt[1]), fontsize=9, ha='right', color='green')
ax1.plot(drone_pos[0], drone_pos[1], 'g*', markersize=18, label='Drone')
ax1.set_xlabel('NED X (north, m)'); ax1.set_ylabel('NED Y (east, m)')
ax1.set_title('NED World Coordinates'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# CARLA
carla_arr = np.array(carla_pts)
ax2.plot(carla_arr[:, 0], carla_arr[:, 1], 'ro-', markersize=8, linewidth=2)
for i, pt in enumerate(carla_pts):
    ax2.annotate(str(i+1), (pt[0], pt[1]), fontsize=9, ha='right', color='red')
ax2.set_xlabel('CARLA X (east, m)'); ax2.set_ylabel('CARLA Y (south, m)')
ax2.set_title('CARLA World Coordinates'); ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

fig.suptitle(f'Pixel Path → World (offset_xy=[{offset_xy[0]:.0f},{offset_xy[1]:.0f}])', fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/test_pixel_path_2_world.png", dpi=120)
plt.close(fig)
print("Saved: test_pixel_path_2_world.png")

print("\nDone!")
