import pandas as pd
import json

df    = pd.read_csv("outputs/detection/trajectories.csv")
lanes = json.load(open("outputs/lanes/lanes.json"))

print("Center X distribution:")
print(df["center_x"].describe())

print(f"\nVehicles with center_x < 700  (left side) : {len(df[df['center_x'] < 700])}")
print(f"Vehicles with center_x >= 700 (right side): {len(df[df['center_x'] >= 700])}")

print("\nLane x boundaries:")
for l in lanes:
    lb = l["left_boundary"]["x_bottom"]
    rb = l["right_boundary"]["x_bottom"]
    print(f"  {l['lane_id']}: x_bottom {lb} to {rb}")