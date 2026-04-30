import cv2
import yaml
import numpy as np
import json
from pathlib import Path
from scipy.signal import find_peaks

CONFIG_PATH = "configs/highway.yaml"
OUTPUT_DIR  = Path("outputs/lanes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_homography(cfg):
    pts      = cfg["calibration"]["reference_points"]
    pix_pts  = np.float32([p["pixel"] for p in pts])
    real_pts = np.float32([p["real"]  for p in pts])
    H, _     = cv2.findHomography(pix_pts, real_pts)
    H_inv, _ = cv2.findHomography(real_pts, pix_pts)
    return H, H_inv

def pixel_to_real(px, py, H):
    pt  = np.float32([[[float(px), float(py)]]])
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])

def real_to_pixel(real_x, real_y, H_inv):
    pt  = np.float32([[[real_x, real_y]]])
    out = cv2.perspectiveTransform(pt, H_inv)
    return (int(out[0][0][0]), int(out[0][0][1]))

# ── Auto lane count detection ────────────────────────────────────────
def auto_detect_lane_count(frame, H,cfg):
    h, w     = frame.shape[:2]
    sample_y = int(h * 0.75)

    # Take wider strip for better averaging
    strip    = frame[sample_y:sample_y + 20, :, :]
    gray     = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    row      = np.mean(gray, axis=0).astype(float)

    # Stronger smoothing to suppress noise
    smoothed = np.convolve(row, np.ones(25)/25, mode="same")

    # Find central barrier using right half of frame
    # Right carriageway is on the RIGHT side of frame
    # Barrier is between x=600 and x=800 based on calibration
    # P1 pixel x = 762 (left edge of right carriageway)
    # So barrier must be to the LEFT of P1
    pts         = cfg["calibration"]["reference_points"]
    p1_x        = pts[0]["pixel"][0]   # left edge of right carriageway
    p2_x        = pts[1]["pixel"][0]   # right edge of right carriageway

    print(f"  Right carriageway pixel range: x={p1_x} to x={p2_x}")

    # Only look for peaks WITHIN the right carriageway region
    road_region = smoothed[p1_x:p2_x]

    # Higher threshold — only strong white lane markings
    threshold = np.mean(road_region) + 1.0 * np.std(road_region)
    peaks, _  = find_peaks(
        road_region,
        height=threshold,
        distance=80          # minimum 80px between peaks
    )

    print(f"  Lane marking peaks found: {len(peaks)}")
    print(f"  Peak positions (relative): {peaks.tolist()}")

    # Lane count = gaps between markings + 1
    # But also include outer edges
    lane_count = len(peaks) + 1

    # Clamp to realistic range
    lane_count = max(2, min(5, lane_count))

    print(f"  Auto-detected lane count = {lane_count}")
    return lane_count
# ── Build lane boundaries from calibration ───────────────────────────
def get_carriageway_width(cfg):
    """
    Auto-compute carriageway width from calibration reference points.
    P1 real x = left edge (0.0), P2 real x = right edge (e.g. 11.0)
    Width = P2.real_x - P1.real_x
    """
    pts   = cfg["calibration"]["reference_points"]
    p1_rx = pts[0]["real"][0]
    p2_rx = pts[1]["real"][0]
    width = abs(p2_rx - p1_rx)
    return round(width, 2)

def validate_width(total_width, num_lanes):
    """
    Validate computed lane width is realistic.
    Standard highway lane = 3.0m to 4.5m
    """
    lane_width = total_width / num_lanes
    if 3.0 <= lane_width <= 4.5:
        print(f"  ✓ Lane width {lane_width:.2f}m is realistic "
              f"(standard = 3.5-3.65m)")
        return True
    else:
        print(f"  ✗ WARNING: Lane width {lane_width:.2f}m is unrealistic")
        print(f"    Expected 3.0-4.5m for standard highway lanes")
        print(f"    Check your reference points P1 and P2 are on")
        print(f"    the outer solid white lines of the carriageway")
        return False
    
def build_lanes_from_calibration(cfg, H_inv, num_lanes):
    total_width = get_carriageway_width(cfg)   # ← auto from calibration
    lane_width  = total_width / num_lanes
    y_near      = 0.5
    y_far       = 95.0

    right_x_positions = [i * lane_width for i in range(num_lanes + 1)]

    lanes = []
    for i in range(num_lanes):
        x_left  = right_x_positions[i]
        x_right = right_x_positions[i + 1]

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
            "width_px": width_px,
            "width_m":  round(lane_width, 2)
        })

    return lanes

# ── Draw lanes on frame ──────────────────────────────────────────────
def draw_lanes(frame, lanes):
    vis     = frame.copy()
    overlay = vis.copy()

    colors = [
        (0,   255, 0),
        (0,   0,   255),
        (255, 165, 0),
        (255, 0,   255),
        (0,   255, 255),
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

        cv2.line(vis,
                 (lb["x_bottom"], lb["y_bottom"]),
                 (lb["x_top"],    lb["y_top"]),
                 (255, 255, 255), 2)
        cv2.line(vis,
                 (rb["x_bottom"], rb["y_bottom"]),
                 (rb["x_top"],    rb["y_top"]),
                 (255, 255, 255), 2)

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

# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Phase 3: Lane Detection (Calibration-based)")
    print("="*50)

    cfg      = load_config(CONFIG_PATH)
    H, H_inv = compute_homography(cfg)
    frame    = cv2.imread("outputs/frames/frame_0000.jpg")
    h, w     = frame.shape[:2]

    print("\n▶ Determining lane count...")
    lane_count_cfg = cfg["lanes"]["count"]

    if str(lane_count_cfg).lower() == "auto":
        print("  Mode: AUTO-DETECTION from video")
        num_lanes = auto_detect_lane_count(frame, H,cfg)
    else:
        num_lanes = int(lane_count_cfg)
        print(f"  Mode: CONFIG value = {num_lanes} lanes")

    print("\n▶ Building lanes from calibration points...")
    total_w = get_carriageway_width(cfg)
    print(f"  Auto-computed carriageway width : {total_w}m")
    print(f"  Lane count                      : {num_lanes}")
    validate_width(total_w, num_lanes)
    lanes = build_lanes_from_calibration(cfg, H_inv, num_lanes)

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