import cv2
import yaml
import numpy as np
import json
from pathlib import Path

CONFIG_PATH = "configs/highway.yaml"
OUTPUT_DIR  = Path("outputs/lanes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_homography(cfg):
    pts     = cfg["calibration"]["reference_points"]
    pix_pts = np.float32([p["pixel"] for p in pts])
    real_pts= np.float32([p["real"]  for p in pts])
    H, _    = cv2.findHomography(pix_pts, real_pts)
    H_inv, _= cv2.findHomography(real_pts, pix_pts)
    return H, H_inv

def real_to_pixel(real_x, real_y, H_inv):
    pt  = np.float32([[[real_x, real_y]]])
    out = cv2.perspectiveTransform(pt, H_inv)
    return (int(out[0][0][0]), int(out[0][0][1]))

def build_lanes_from_calibration(cfg, H_inv, frame_height):
    """
    Use homography to place exact lane boundaries.
    Right carriageway: 3 lanes, each 3.67m wide (11m / 3)
    Left carriageway:  3 lanes, mirrored on other side
    We define boundaries at real-world x = 0, 3.67, 7.33, 11.0
    at two depths: y=0 (near) and y=20 (far)
    """
    # READ FROM CONFIG (generalised)
    num_lanes   = cfg["lanes"]["count"]
    total_width = cfg["lanes"]["total_width_m"]
    lane_width  = total_width / num_lanes
    y_near = 0.5
    y_far  = 35.0              # slightly in from top

    # Right carriageway lane boundary x positions in real world
    right_x_positions = [i * lane_width for i in range(4)]  # 0,3.67,7.33,11

    lanes = []
    # READ FROM CONFIG (generalised)
    for i in range(num_lanes):
        x_left  = right_x_positions[i]
        x_right = right_x_positions[i + 1]

        # Convert real-world corners to pixel coordinates
        lb_bottom = real_to_pixel(x_left,  y_near, H_inv)
        lb_top    = real_to_pixel(x_left,  y_far,  H_inv)
        rb_bottom = real_to_pixel(x_right, y_near, H_inv)
        rb_top    = real_to_pixel(x_right, y_far,  H_inv)

        width_px = abs(rb_bottom[0] - lb_bottom[0])

        lanes.append({
            "lane_id":       f"right_carriageway_lane{i+1}",
            "carriageway":   "right_carriageway",
            "lane_index":    i + 1,
            "direction":     "toward_camera",
            "left_boundary": {
                "x_bottom": lb_bottom[0], "y_bottom": lb_bottom[1],
                "x_top":    lb_top[0],    "y_top":    lb_top[1]
            },
            "right_boundary": {
                "x_bottom": rb_bottom[0], "y_bottom": rb_bottom[1],
                "x_top":    rb_top[0],    "y_top":    rb_top[1]
            },
            "width_px":  width_px,
            "width_m":   round(lane_width, 2)
        })

    return lanes

def draw_lanes(frame, lanes):
    vis     = frame.copy()
    overlay = vis.copy()

    colors = [
        (0,   255, 0),
        (0,   0,   255),
        (255, 165, 0),
    ]

    for i, lane in enumerate(lanes):
        lb  = lane["left_boundary"]
        rb  = lane["right_boundary"]
        pts = np.array([
            [lb["x_bottom"], lb["y_bottom"]],
            [lb["x_top"],    lb["y_top"]],
            [rb["x_top"],    rb["y_top"]],
            [rb["x_bottom"], rb["y_bottom"]]
        ], dtype=np.int32)

        color = colors[i % len(colors)]
        cv2.fillPoly(overlay, [pts], color)

        # Draw boundary lines
        cv2.line(vis,
                 (lb["x_bottom"], lb["y_bottom"]),
                 (lb["x_top"],    lb["y_top"]),
                 (255, 255, 255), 2)
        cv2.line(vis,
                 (rb["x_bottom"], rb["y_bottom"]),
                 (rb["x_top"],    rb["y_top"]),
                 (255, 255, 255), 2)

        # Lane label at center
        cx = (lb["x_bottom"] + rb["x_bottom"]) // 2
        cy = (lb["y_bottom"] + rb["y_bottom"]) // 2 - 60
        cv2.putText(vis, lane["lane_id"],
                    (cx - 80, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"{lane['width_m']}m",
                    (cx - 20, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 2)

    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
    return vis

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Phase 3: Lane Detection (Calibration-based)")
    print("="*50)

    cfg         = load_config(CONFIG_PATH)
    H, H_inv    = compute_homography(cfg)
    frame       = cv2.imread("outputs/frames/frame_0000.jpg")
    h, w        = frame.shape[:2]

    print("\n▶ Building lanes from calibration points...")
    lanes = build_lanes_from_calibration(cfg, H_inv, h)

    print(f"  Total lanes: {len(lanes)}")
    for lane in lanes:
        print(f"    {lane['lane_id']:30s}  "
              f"width={lane['width_m']}m  "
              f"width_px={lane['width_px']}px")

    print("\n▶ Saving visualisation...")
    vis      = draw_lanes(frame, lanes)
    vis_path = str(OUTPUT_DIR / "lanes_detected.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"  ✓ Saved: {vis_path}")

    json_path = OUTPUT_DIR / "lanes.json"
    with open(json_path, "w") as f:
        json.dump(lanes, f, indent=2)
    print(f"  ✓ Lane data saved: {json_path}")

    print("\n" + "="*50)
    print("  Phase 3 Complete!")
    print("="*50)