import cv2
import yaml
import numpy as np
from pathlib import Path

VIDEO_PATH = "data/videos/highway.mp4"
OUTPUT_DIR = "outputs/frames"

# ── Step 1: Inspect video properties ────────────────────────────────
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
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration
    }

# ── Step 2: Extract key frames ───────────────────────────────────────
def extract_frames(path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap          = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save frame 0, 25%, 50%, 75%, last
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
def pick_reference_points(frame_path):
    """
    Click exactly 4 points on the road forming a rectangle.
    Best points: lane marking corners whose real distance you know.
    Standard lane width = 3.5m (UK highway standard)
    Press Q when done.
    """
    img    = cv2.imread(frame_path)
    clone  = img.copy()
    points = []

    instructions = [
    "Click P1: Bottom-LEFT edge of right carriageway (near camera)",
    "Click P2: Bottom-RIGHT edge of right carriageway (near camera, 11m away)",
    "Click P3: Top-RIGHT edge of right carriageway (far from camera)",
    "Click P4: Top-LEFT edge of right carriageway (far from camera)",
]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(img, f"P{len(points)}({x},{y})",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)
            if len(points) < 4:
                print(f"  ✓ P{len(points)} set at ({x}, {y})")
                print(f"  → Next: {instructions[len(points)]}")
            else:
                print(f"  ✓ P4 set at ({x}, {y})")
                print("\n  All 4 points selected! Press Q to close.")
            cv2.imshow("Pick 4 Reference Points — Press Q when done", img)

    print("\n" + "=" * 50)
    print("  REFERENCE POINT PICKER")
    print("=" * 50)
    print(f"  → Start: {instructions[0]}")
    print("  Tip: Pick points on ONE side of the highway")
    print("       (left carriageway only, ignore right side)")
    print("=" * 50 + "\n")

    cv2.imshow("Pick 4 Reference Points — Press Q when done", img)
    cv2.setMouseCallback(
        "Pick 4 Reference Points — Press Q when done", on_click
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

# ── Step 4: Auto-generate config.yaml ───────────────────────────────
def generate_config(video_info, ref_points, output_path="configs/highway.yaml"):
    """
    Real-world coordinates assumption (UK highway):
    P1 = origin (0, 0)
    P2 = one lane width to the right (3.5, 0)
    P3 = one lane width right + 20m ahead (3.5, 20)
    P4 = 20m ahead (0, 20)
    Adjust the 20.0 value if you know the actual road length in view.
    """
    config = {
        "video": {
            "path": "data/videos/highway.mp4",
            "fps": video_info["fps"],
            "resolution": [video_info["width"], video_info["height"]],
            "total_frames": video_info["total_frames"],
            "duration": round(video_info["duration"], 2)
        },
        "calibration": {
            "reference_points": [
                {"pixel": list(ref_points[0]), "real": [0.0,  0.0]},   # P1 near-left
                {"pixel": list(ref_points[1]), "real": [11.0, 0.0]},   # P2 near-right (11m = 3 lanes)
                {"pixel": list(ref_points[2]), "real": [11.0, 20.0]},  # P3 far-right
                {"pixel": list(ref_points[3]), "real": [0.0,  20.0]},  # P4 far-left
            ]
        },
        "detection": {
            "yolo_model": "yolov8n.pt",
            "confidence_threshold": 0.45,
            "classes": ["car", "truck", "bus", "motorcycle"]
        },
        "lanes": {
          "count":       "auto",
          "carriageway": "right"
      },
        "vissim": {
            "version": 10,
            "simulation_duration": 3600,
            "wiedemann_model": 99
        }
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n  ✓ Config saved to: {output_path}")
    print("\n  Config contents:")
    print("-" * 50)
    with open(output_path) as f:
        print(f.read())

# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n▶ Step 1: Inspecting video...")
    info = inspect_video(VIDEO_PATH)
    if info is None:
        exit(1)

    print("\n▶ Step 2: Extracting reference frames...")
    frames = extract_frames(VIDEO_PATH, OUTPUT_DIR)

    print("\n▶ Step 3: Pick 4 reference points on the road.")
    print("  Opening frame 0 for point selection...")
    points = pick_reference_points(frames[0])

    if len(points) != 4:
        print(f"ERROR: Need exactly 4 points, got {len(points)}")
        exit(1)

    print("\n▶ Step 4: Generating config.yaml...")
    generate_config(info, points)

    print("\n" + "=" * 50)
    print("  Phase 1 Complete!")
    print("  outputs/frames/  → reference frames saved")
    print("  configs/highway.yaml  → config generated")
    print("=" * 50)