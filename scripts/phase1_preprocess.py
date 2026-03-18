import cv2
import yaml
import numpy as np
from pathlib import Path

VIDEO_PATH = "data/videos/highway.mp4"
OUTPUT_DIR = "outputs/frames"

# ── Road width lookup table ──────────────────────────────────────────
ROAD_WIDTHS = {
    "motorway_2lane":  7.0,    # 2 lanes x 3.5m
    "motorway_3lane":  11.0,   # 3 lanes x 3.67m  ← UK standard
    "motorway_4lane":  14.0,   # 4 lanes x 3.5m
    "motorway_5lane":  17.5,   # 5 lanes x 3.5m
    "urban_2lane":     6.0,    # 2 lanes x 3.0m
    "urban_3lane":     9.0,    # 3 lanes x 3.0m
    "urban_4lane":     12.0,   # 4 lanes x 3.0m
    "rural_2lane":     7.3,    # 2 lanes x 3.65m
    "rural_3lane":     10.95,  # 3 lanes x 3.65m
}

# ── Step 1: Inspect video ────────────────────────────────────────────
def inspect_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {path}")
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    print("=" * 50)
    print(f"  Video      : {path}")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {duration:.2f} seconds")
    print("=" * 50)

    cap.release()
    return {
        "fps": fps, "width": width, "height": height,
        "total_frames": total_frames, "duration": duration
    }

# ── Step 2: Extract reference frames ────────────────────────────────
def extract_frames(path, output_dir):
    import shutil
    # Clear old frames before extracting new ones
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
        print(f"  Cleared old frames from {output_dir}")
    out.mkdir(parents=True, exist_ok=True)
    cap          = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    targets = [
        0,
        total_frames // 4,
        total_frames // 2,
        (3 * total_frames) // 4,
        total_frames - 1
    ]

    saved = []
    for t in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, frame = cap.read()
        if ret:
            out_path = f"{output_dir}/frame_{t:04d}.jpg"
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            print(f"  Saved: {out_path}")

    cap.release()
    return saved

# ── Step 3: Interactive reference point picker ───────────────────────
def pick_reference_points(frame_path, road_width):
    img    = cv2.imread(frame_path)
    points = []

    instructions = [
        "Click P1: Bottom-LEFT  solid white line (near camera)",
        "Click P2: Bottom-RIGHT solid white line (near camera)",
        f"Click P3: Top-RIGHT    above P2 position (far from camera)",
        f"Click P4: Top-LEFT     above P1 position (far from camera)",
    ]

    print("\n" + "=" * 50)
    print("  REFERENCE POINT PICKER")
    print("=" * 50)
    print(f"  Road width : {road_width}m (auto-set from road_type)")
    print(f"  → Start: {instructions[0]}")
    print("  TIP: P3 and P4 may fall slightly outside white lines")
    print("       due to perspective — this is correct and expected")
    print("=" * 50 + "\n")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(img, f"P{len(points)}({x},{y})",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)
            print(f"  ✓ P{len(points)} set at ({x}, {y})")
            if len(points) < 4:
                print(f"  → Next: {instructions[len(points)]}")
            else:
                print(f"\n  All 4 points selected! Press Q to close.")
            cv2.imshow("Pick 4 Reference Points — Press Q when done", img)

    cv2.imshow("Pick 4 Reference Points — Press Q when done", img)
    cv2.setMouseCallback(
        "Pick 4 Reference Points — Press Q when done", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points

# ── Step 4: Generate config ──────────────────────────────────────────
def generate_config(video_info, ref_points, road_type,
                    output_path="configs/highway.yaml"):

    road_width = ROAD_WIDTHS.get(road_type, 11.0)

    # Real-world coords auto-filled from road_type
    # P1 = origin, P2 = road_width away, P3/P4 = 20m depth
    config = {
        "video": {
            "path":         VIDEO_PATH,
            "fps":          video_info["fps"],
            "resolution":   [video_info["width"], video_info["height"]],
            "total_frames": video_info["total_frames"],
            "duration":     round(video_info["duration"], 2)
        },
        "road_type": road_type,
        "calibration": {
            "reference_points": [
                {"pixel": list(ref_points[0]),
                 "real":  [0.0,        0.0 ]},
                {"pixel": list(ref_points[1]),
                 "real":  [road_width, 0.0 ]},   # ← auto from road_type
                {"pixel": list(ref_points[2]),
                 "real":  [road_width, 20.0]},   # ← auto from road_type
                {"pixel": list(ref_points[3]),
                 "real":  [0.0,        20.0]},
            ]
        },
        "detection": {
            "yolo_model":           "yolov8n.pt",
            "confidence_threshold": 0.45,
            "classes": ["car", "truck", "bus", "motorcycle"]
        },
        "lanes": {
            "count":       "auto",
            "carriageway": "right"
        },
        "vissim": {
            "version":                  10,
            "simulation_duration":      3600,
            "wiedemann_model":          99,
            "central_barrier_width_m":  1.0,
            "create_both_carriageways": True
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n  ✓ Config saved: {output_path}")
    print(f"  ✓ Road type   : {road_type}")
    print(f"  ✓ Road width  : {road_width}m (auto-assigned to P2/P3)")
    print("\n  Reference points:")
    for i, (px, rx) in enumerate(zip(
        ref_points,
        [[0.0, 0.0], [road_width, 0.0],
         [road_width, 20.0], [0.0, 20.0]]
    )):
        print(f"    P{i+1}: pixel={list(px)}  real={rx}")

# ── Step 5: Verify homography ────────────────────────────────────────
def verify_homography(cfg_path):
    import cv2
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pts      = cfg["calibration"]["reference_points"]
    pix_pts  = np.float32([p["pixel"] for p in pts])
    real_pts = np.float32([p["real"]  for p in pts])
    H, _     = cv2.findHomography(pix_pts, real_pts)

    test1 = cv2.perspectiveTransform(
        np.float32([[pix_pts[0]]]), H)[0][0]
    test2 = cv2.perspectiveTransform(
        np.float32([[pix_pts[1]]]), H)[0][0]

    print(f"\n  Homography verification:")
    print(f"  P1 pixel → real: {test1}  (expected ~0, 0)")
    print(f"  P2 pixel → real: {test2}  "
          f"(expected ~{real_pts[1][0]}, 0)")

    width_ok = abs(test2[0] - real_pts[1][0]) < 0.1
    print(f"  Width check: {'✓ PASS' if width_ok else '✗ FAIL'}")
    return width_ok

# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Read road_type from existing config if available ──────────────
    try:
        with open("configs/highway.yaml", encoding="utf-8") as f:
            existing = yaml.safe_load(f)
        road_type  = existing.get("road_type", "motorway_3lane")
        video_path = existing["video"]["path"]
    except Exception:
        road_type  = "motorway_3lane"
        video_path = VIDEO_PATH

    print("\n" + "="*50)
    print("  Phase 1: Video Inspection + Calibration")
    print("="*50)
    print(f"\n  Road type : {road_type}")
    print(f"  Width     : {ROAD_WIDTHS.get(road_type, 11.0)}m")
    print(f"\n  To change road type, update 'road_type' in")
    print(f"  configs/highway.yaml before running phase1.")
    print(f"\n  Available road types:")
    for k, v in ROAD_WIDTHS.items():
        marker = " ← current" if k == road_type else ""
        print(f"    {k:20s} = {v}m{marker}")

    print("\n▶ Step 1: Inspecting video...")
    info = inspect_video(video_path)
    if info is None:
        exit(1)

    print("\n▶ Step 2: Extracting reference frames...")
    frames = extract_frames(video_path, OUTPUT_DIR)

    print("\n▶ Step 3: Pick 4 reference points...")
    road_width = ROAD_WIDTHS.get(road_type, 11.0)
    points     = pick_reference_points(frames[0], road_width)

    if len(points) != 4:
        print(f"ERROR: Need 4 points, got {len(points)}")
        exit(1)

    print("\n▶ Step 4: Generating config...")
    generate_config(info, points, road_type)

    print("\n▶ Step 5: Verifying homography...")
    verify_homography("configs/highway.yaml")

    print("\n" + "="*50)
    print("  Phase 1 Complete!")
    print("  outputs/frames/     → reference frames")
    print("  configs/highway.yaml → config generated")
    print("="*50)
