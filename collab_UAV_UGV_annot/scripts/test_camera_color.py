#!/usr/bin/env python3
"""Quick camera color test — spawn FPV camera, capture one frame, save and exit.

Usage:
  conda activate carlaAir
  python collab_UAV_UGV/scripts/test_camera_color.py
"""

import sys, time
import carla, cv2
import numpy as np


def main():
    cl = carla.Client("127.0.0.1", 2000)
    cl.set_timeout(10.0)
    world = cl.get_world()

    # Find the Mini Cooper
    car = None
    for v in world.get_actors().filter("vehicle.*"):
        tid = v.type_id.lower()
        if "carlacola" in tid or "hgv" in tid:
            continue
        if "mini" in tid:
            car = v
            break
    if car is None:
        print("ERROR: Mini Cooper not found — run setup_scene.py first")
        return 1

    print(f"Car: {car.type_id}")

    # Spawn FPV camera with color-tuning attributes
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "540")
    bp.set_attribute("fov", "110")
    bp.set_attribute("temp", "5500")
    bp.set_attribute("tint", "0")
    bp.set_attribute("exposure_mode", "manual")
    bp.set_attribute("gamma", "2.2")
    tf = carla.Transform(carla.Location(x=1.8, z=1.3))
    cam = world.spawn_actor(bp, tf, attach_to=car)

    frame = [None]

    def cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        frame[0] = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]

    cam.listen(cb)
    time.sleep(1.0)  # wait for first frame

    if frame[0] is None:
        print("ERROR: no frame received")
        cam.stop()
        cam.destroy()
        return 1

    out = "/home/zsn/VLN/carlaAir_experiments/collab_UAV_UGV/runs/test_fpv_color.jpg"
    cv2.imwrite(out, frame[0])
    print(f"Saved: {out}")
    print(f"Camera attributes: temp=5500  tint=0  exposure_mode=manual  gamma=2.2")
    print(f"  To adjust: edit bp.set_attribute lines in this script, re-run, compare images.")

    cam.stop()
    cam.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
