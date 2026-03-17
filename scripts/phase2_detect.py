import cv2
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────
CONFIG_PATH  = "configs/highway.yaml"
OUTPUT_DIR   = Path("outputs/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO class IDs we care about (COCO dataset)
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

# ── Main detection + tracking ─────────────────────────────────────────
def run_detection(cfg):
    video_path = cfg["video"]["path"]
    conf_thresh = cfg["detection"]["confidence_threshold"]
    model_name  = cfg["detection"]["yolo_model"]

    print(f"  Loading YOLO model: {model_name}")
    model = YOLO(model_name)  # auto-downloads yolov8n.pt if not present

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps          = cfg["video"]["fps"]
    width        = cfg["video"]["resolution"][0]
    height       = cfg["video"]["resolution"][1]
    total_frames = cfg["video"]["total_frames"]

    # ── Output video writer ───────────────────────────────────────────
    out_video_path = str(OUTPUT_DIR / "detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # ── Trajectory storage ────────────────────────────────────────────
    records = []
    frame_idx = 0

    print(f"  Processing {total_frames} frames...")
    print("  This may take a few minutes on CPU — please wait.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO with ByteTrack
        results = model.track(
            frame,
            persist=True,           # maintains track IDs across frames
            tracker="bytetrack.yaml",
            conf=conf_thresh,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )

        result = results[0]

        # ── Extract detections ────────────────────────────────────────
        if result.boxes is not None and result.boxes.id is not None:
            boxes   = result.boxes.xyxy.cpu().numpy()    # x1,y1,x2,y2
            ids     = result.boxes.id.cpu().numpy()      # track IDs
            classes = result.boxes.cls.cpu().numpy()     # class IDs
            confs   = result.boxes.conf.cpu().numpy()    # confidence

            for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2   # center x
                cy = (y1 + y2) / 2   # center y
                cls_id = int(cls)

                records.append({
                    "frame":      frame_idx,
                    "time_s":     round(frame_idx / fps, 3),
                    "vehicle_id": int(tid),
                    "class":      VEHICLE_CLASSES.get(cls_id, "unknown"),
                    "bbox_x1":    round(x1, 1),
                    "bbox_y1":    round(y1, 1),
                    "bbox_x2":    round(x2, 1),
                    "bbox_y2":    round(y2, 1),
                    "center_x":   round(cx, 1),
                    "center_y":   round(cy, 1),
                    "confidence": round(float(conf), 3)
                })

                # ── Draw on frame ─────────────────────────────────────
                color = {
                    "car":        (0, 255, 0),
                    "truck":      (0, 0, 255),
                    "bus":        (255, 165, 0),
                    "motorcycle": (255, 0, 255)
                }.get(VEHICLE_CLASSES.get(cls_id, ""), (200, 200, 200))

                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                cv2.putText(frame,
                            f"ID{int(tid)} {VEHICLE_CLASSES.get(cls_id,'')}",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Progress every 30 frames
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total_frames} "
                  f"({100*frame_idx//total_frames}%) — "
                  f"{len(records)} detections so far")

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # ── Save trajectories CSV ─────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "trajectories.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n  ✓ Annotated video saved : {out_video_path}")
    print(f"  ✓ Trajectories saved    : {csv_path}")
    print(f"  ✓ Total detections      : {len(records)}")
    print(f"  ✓ Unique vehicles       : {df['vehicle_id'].nunique()}")
    print(f"\n  Vehicle breakdown:")
    print(df.groupby("class")["vehicle_id"].nunique().to_string())

    return df

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Phase 2: YOLO Detection + ByteTrack")
    print("="*50)

    cfg = load_config(CONFIG_PATH)
    df  = run_detection(cfg)

    print("\n" + "="*50)
    print("  Phase 2 Complete!")
    print("  Check outputs/detection/ for results")
    print("="*50)