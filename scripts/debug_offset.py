import cv2
import numpy as np
import yaml
import json

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config("configs/highway.yaml")
pts      = cfg["calibration"]["reference_points"]
pix_pts  = np.float32([p["pixel"] for p in pts])
real_pts = np.float32([p["real"]  for p in pts])
H, _     = cv2.findHomography(pix_pts, real_pts)
H_inv, _ = cv2.findHomography(real_pts, pix_pts)

def pixel_to_real(px, py):
    pt  = np.float32([[[float(px), float(py)]]])
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])

def real_to_pixel(rx, ry):
    pt  = np.float32([[[float(rx), float(ry)]]])
    out = cv2.perspectiveTransform(pt, H_inv)
    return int(out[0][0][0]), int(out[0][0][1])

# Check where key pixel x positions map to in real world
print("Pixel x → Real world x (at y=595, bottom of frame):")
for px in [100, 200, 300, 400, 500, 600, 650, 700, 750]:
    rx, ry = pixel_to_real(px, 595)
    print(f"  pixel x={px:4d} → real x={rx:7.2f}m")

print()

# Check right carriageway boundary
print("Right carriageway starts at pixel x=757:")
rx, ry = pixel_to_real(757, 595)
print(f"  → real x = {rx:.2f}m")

# Where should left carriageway right edge be?
# Visually it looks like x~600-650 in pixel space
print()
print("Left carriageway right edge estimate (pixel x=620):")
rx, ry = pixel_to_real(620, 595)
print(f"  → real x = {rx:.2f}m")

print()
print("Left carriageway left edge estimate (pixel x=150):")
rx, ry = pixel_to_real(150, 595)
print(f"  → real x = {rx:.2f}m")

# Visualize on frame
frame = cv2.imread("outputs/frames/frame_0000.jpg")
# Draw vertical lines at key x positions
for px, color, label in [
    (150,  (0,255,255), "left edge"),
    (350,  (0,200,255), "L-lane2"),
    (550,  (0,150,255), "L-lane3"),
    (620,  (255,255,0), "barrier L"),
    (757,  (0,255,0),   "barrier R"),
    (892,  (0,255,0),   "R-lane2"),
    (1026, (0,255,0),   "R-lane3"),
    (1159, (0,255,0),   "right edge"),
]:
    cv2.line(frame, (px, 0), (px, frame.shape[0]), color, 2)
    cv2.putText(frame, label, (px-30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite("outputs/frames/frame_offset_debug.jpg", frame)
print("\nSaved: outputs/frames/frame_offset_debug.jpg")