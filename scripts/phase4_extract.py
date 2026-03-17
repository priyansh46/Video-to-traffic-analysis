import cv2
import yaml
import numpy as np
import pandas as pd
import json
from pathlib import Path

CONFIG_PATH = "configs/highway.yaml"
OUTPUT_DIR  = Path("outputs/parameters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_homography(cfg):
    pts      = cfg["calibration"]["reference_points"]
    pix_pts  = np.float32([p["pixel"] for p in pts])
    real_pts = np.float32([p["real"]  for p in pts])
    H, _     = cv2.findHomography(pix_pts, real_pts)
    return H

def pixel_to_real(px, py, H):
    pt  = np.float32([[[float(px), float(py)]]])
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])

# ── Assign each vehicle detection to a lane ──────────────────────────
def assign_to_lanes(df, lanes):
    df = df.copy()
    df["lane_id"] = "unknown"

    def interp_x(b, y):
        if b["y_bottom"] == b["y_top"]:
            return b["x_bottom"]
        t = (y - b["y_bottom"]) / (b["y_top"] - b["y_bottom"])
        return b["x_bottom"] + t * (b["x_top"] - b["x_bottom"])

    def get_lane(cx, cy):
        for lane in lanes:
            lb = lane["left_boundary"]
            rb = lane["right_boundary"]

            # Check if cy is within vertical range of lane
            y_min = min(lb["y_top"],    rb["y_top"])
            y_max = max(lb["y_bottom"], rb["y_bottom"])

            if not (y_min <= cy <= y_max):
                continue

            x_left  = interp_x(lb, cy)
            x_right = interp_x(rb, cy)

            # Ensure left < right
            x_left, x_right = min(x_left, x_right), max(x_left, x_right)

            if x_left <= cx <= x_right:
                return lane["lane_id"]
        return "unknown"

    # Apply lane assignment to every row efficiently
    df["lane_id"] = df.apply(
        lambda row: get_lane(row["center_x"], row["center_y"]), axis=1
    )
    return df

# ── Compute speed per vehicle using homography ───────────────────────
def compute_speeds(df, H, fps):
    speeds = []
    vehicle_ids = df["vehicle_id"].unique()

    for vid in vehicle_ids:
        vdf = df[df["vehicle_id"] == vid].sort_values("frame")
        if len(vdf) < 2:
            continue

        frame_speeds = []
        rows = vdf.to_dict("records")

        for i in range(1, len(rows)):
            px1, py1 = rows[i-1]["center_x"], rows[i-1]["center_y"]
            px2, py2 = rows[i  ]["center_x"], rows[i  ]["center_y"]

            rx1, ry1 = pixel_to_real(px1, py1, H)
            rx2, ry2 = pixel_to_real(px2, py2, H)

            dist_m   = np.sqrt((rx2-rx1)**2 + (ry2-ry1)**2)
            dt_s     = 1.0 / fps
            speed_ms = dist_m / dt_s
            speed_kmh= speed_ms * 3.6

            # Filter unrealistic speeds (0–200 km/h)
            if 20 < speed_kmh < 160:
                frame_speeds.append({
                    "vehicle_id": vid,
                    "frame":      rows[i]["frame"],
                    "speed_kmh":  round(speed_kmh, 2),
                    "lane_id":    rows[i]["lane_id"],
                    "class":      rows[i]["class"]
                })

        speeds.extend(frame_speeds)

    return pd.DataFrame(speeds)

# ── Compute flow rate per lane (vehicles per hour) ───────────────────
def compute_flow(df, fps, total_frames):
    duration_s   = total_frames / fps
    duration_hr  = duration_s / 3600

    flow_results = []
    for lane_id in df["lane_id"].unique():
        if lane_id == "unknown":
            continue
        lane_df     = df[df["lane_id"] == lane_id]
        unique_vehs = lane_df["vehicle_id"].nunique()
        flow_vph    = unique_vehs / duration_hr

        flow_results.append({
            "lane_id":        lane_id,
            "vehicle_count":  unique_vehs,
            "duration_s":     round(duration_s, 2),
            "flow_vph":       round(flow_vph, 1)
        })

    return pd.DataFrame(flow_results)

# ── Compute headway per lane ─────────────────────────────────────────
def compute_headways(df, fps):
    """
    Time headway: time between consecutive vehicles
    crossing the same point in the same lane.
    We use a virtual detector line at y = bottom of lane.
    """
    headway_results = []

    for lane_id in df["lane_id"].unique():
        if lane_id == "unknown":
            continue

        lane_df    = df[df["lane_id"] == lane_id].copy()
        # Get first appearance frame for each vehicle in this lane
        first_seen = (lane_df.groupby("vehicle_id")["frame"]
                             .min()
                             .sort_values())

        times = first_seen.values / fps
        if len(times) < 2:
            continue

        headways = np.diff(times)
        headways = headways[headways > 0]

        if len(headways) == 0:
            continue

        headway_results.append({
            "lane_id":         lane_id,
            "mean_headway_s":  round(float(np.mean(headways)), 3),
            "min_headway_s":   round(float(np.min(headways)),  3),
            "max_headway_s":   round(float(np.max(headways)),  3),
        })

    return pd.DataFrame(headway_results)

# ── Compute vehicle composition per lane ────────────────────────────
def compute_composition(df):
    results = []
    for lane_id in df["lane_id"].unique():
        if lane_id == "unknown":
            continue
        lane_df = df[df["lane_id"] == lane_id]
        comp    = (lane_df.groupby("class")["vehicle_id"]
                          .nunique())
        total   = comp.sum()
        if total == 0:
            continue
        row = {"lane_id": lane_id, "total_vehicles": int(total)}
        for cls in ["car", "truck", "bus", "motorcycle"]:
            count       = int(comp.get(cls, 0))
            row[cls]    = count
            row[f"pct_{cls}"] = round(count / total, 3)
        results.append(row)
    return pd.DataFrame(results)

# ── Build final VISSIM-ready parameters ─────────────────────────────
def build_vissim_params(lanes_data, flow_df, speed_df,
                        headway_df, comp_df):
    records = []
    for lane in lanes_data:
        lid = lane["lane_id"]

        flow_row    = (flow_df[flow_df["lane_id"] == lid]
                       .iloc[0] if lid in flow_df["lane_id"].values
                       else None)
        speed_rows  = (speed_df[speed_df["lane_id"] == lid]
                       if lid in speed_df["lane_id"].values
                       else pd.DataFrame())
        hw_row      = (headway_df[headway_df["lane_id"] == lid]
                       .iloc[0] if lid in headway_df["lane_id"].values
                       else None)
        comp_row    = (comp_df[comp_df["lane_id"] == lid]
                       .iloc[0] if lid in comp_df["lane_id"].values
                       else None)

        record = {
            "lane_id":        lid,
            "carriageway":    lane["carriageway"],
            "lane_index":     lane["lane_index"],
            "width_m":        lane["width_m"],
            # Geometry
            "start_x_px":     lane["left_boundary"]["x_bottom"],
            "start_y_px":     lane["left_boundary"]["y_bottom"],
            "end_x_px":       lane["left_boundary"]["x_top"],
            "end_y_px":       lane["left_boundary"]["y_top"],
            # Flow
            "flow_vph":       round(flow_row["flow_vph"], 1)
                              if flow_row is not None else 0,
            "vehicle_count":  int(flow_row["vehicle_count"])
                              if flow_row is not None else 0,
            # Speed
            "mean_speed_kmh": round(speed_rows["speed_kmh"].mean(), 2)
                              if len(speed_rows) > 0 else 0,
            "std_speed_kmh":  round(speed_rows["speed_kmh"].std(),  2)
                              if len(speed_rows) > 0 else 0,
            "min_speed_kmh":  round(speed_rows["speed_kmh"].min(),  2)
                              if len(speed_rows) > 0 else 0,
            "max_speed_kmh":  round(speed_rows["speed_kmh"].max(),  2)
                              if len(speed_rows) > 0 else 0,
            # Headway
            "mean_headway_s": round(hw_row["mean_headway_s"], 3)
                              if hw_row is not None else 0,
            # Composition
            "pct_car":        round(comp_row["pct_car"],   3)
                              if comp_row is not None else 0,
            "pct_truck":      round(comp_row["pct_truck"], 3)
                              if comp_row is not None else 0,
            "pct_bus":        round(comp_row["pct_bus"],   3)
                              if comp_row is not None else 0,
        }
        records.append(record)

    return pd.DataFrame(records)

# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Phase 4: Traffic Parameter Extraction")
    print("="*50)

    cfg          = load_config(CONFIG_PATH)
    H            = compute_homography(cfg)
    fps          = cfg["video"]["fps"]
    total_frames = cfg["video"]["total_frames"]

    print("\n▶ Loading trajectories and lane data...")
    df    = pd.read_csv("outputs/detection/trajectories.csv")
    lanes = json.load(open("outputs/lanes/lanes.json"))
    print(f"  Trajectories: {len(df)} rows")
    print(f"  Lanes:        {len(lanes)}")

    print("\n▶ Assigning vehicles to lanes...")
    df = assign_to_lanes(df, lanes)
    assigned = df[df["lane_id"] != "unknown"]
    print(f"  Assigned : {len(assigned)} / {len(df)} detections")
    print(f"  Per lane :")
    print(df["lane_id"].value_counts().to_string())

    print("\n▶ Computing speeds...")
    speed_df = compute_speeds(df, H, fps)
    print(f"  Speed records: {len(speed_df)}")
    if len(speed_df) > 0:
        print(f"  Mean speed   : "
              f"{speed_df['speed_kmh'].mean():.1f} km/h")
        print(f"  Speed range  : "
              f"{speed_df['speed_kmh'].min():.1f} – "
              f"{speed_df['speed_kmh'].max():.1f} km/h")

    print("\n▶ Computing flow rates...")
    flow_df = compute_flow(df, fps, total_frames)
    print(flow_df.to_string(index=False))

    print("\n▶ Computing headways...")
    headway_df = compute_headways(df, fps)
    print(headway_df.to_string(index=False))

    print("\n▶ Computing vehicle composition...")
    comp_df = compute_composition(df)
    print(comp_df[["lane_id","total_vehicles",
                   "pct_car","pct_truck","pct_bus"]]
          .to_string(index=False))

    print("\n▶ Building VISSIM-ready parameters...")
    vissim_df = build_vissim_params(lanes, flow_df, speed_df,
                                    headway_df, comp_df)

    # Save all outputs
    speed_df.to_csv(  OUTPUT_DIR / "speeds.csv",     index=False)
    flow_df.to_csv(   OUTPUT_DIR / "flow.csv",        index=False)
    headway_df.to_csv(OUTPUT_DIR / "headways.csv",    index=False)
    comp_df.to_csv(   OUTPUT_DIR / "composition.csv", index=False)
    vissim_df.to_csv( OUTPUT_DIR / "vissim_params.csv", index=False)

    print(f"\n  ✓ speeds.csv       saved")
    print(f"  ✓ flow.csv         saved")
    print(f"  ✓ headways.csv     saved")
    print(f"  ✓ composition.csv  saved")
    print(f"  ✓ vissim_params.csv saved")

    print("\n  Final VISSIM parameters summary:")
    print("-"*50)
    print(vissim_df[["lane_id", "flow_vph",
                     "mean_speed_kmh", "mean_headway_s",
                     "pct_car", "pct_truck"]].to_string(index=False))

    print("\n" + "="*50)
    print("  Phase 4 Complete!")
    print("  Check outputs/parameters/ for all CSVs")
    print("="*50)