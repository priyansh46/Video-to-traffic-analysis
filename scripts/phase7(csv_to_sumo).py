import pandas as pd
import subprocess
import os

CSV_PATH = "/mnt/c/temp/trajectories_xy.csv"
OUT_DIR  = "/home/priyansh/highway_v2x/simulations"

df = pd.read_csv(CSV_PATH)
df.columns = [c.replace("\\","_") for c in df.columns]
df["SIMSEC"] = df["SIMSEC"].round(1)
df["NO"] = df["NO"].astype(int)

# Auto-detect from data
links      = sorted(df["LANE_LINK_NO"].unique())
num_lanes  = int(df["LANE_INDEX"].max())
x_min      = df["X"].min()
x_max      = df["X"].max()
y_min      = df["Y"].min()
y_max      = df["Y"].max()
speed_ms   = float(df["SPEED"].max()) / 3.6
duration   = df["SIMSEC"].max()

print(f"Auto-detected:")
print(f"  Links     : {links}")
print(f"  Lanes     : {num_lanes}")
print(f"  X range   : {x_min:.1f} to {x_max:.1f}")
print(f"  Y range   : {y_min:.1f} to {y_max:.1f}")
print(f"  Max speed : {speed_ms:.1f} m/s")
print(f"  Duration  : {duration}s")

# Separate right and left carriageways by Y position
# Right carriageway = negative Y, Left carriageway = positive Y
link_x = df.groupby("LANE_LINK_NO")["X"].mean()
right_links = sorted(link_x[link_x > 0].index.tolist())
left_links  = sorted(link_x[link_x < 0].index.tolist())
right_lanes = len(right_links)
left_lanes  = len(left_links)
print(f"  Right carriageway links: {right_links} ({right_lanes} lanes)")
print(f"  Left carriageway links : {left_links}  ({left_lanes} lanes)")

# Write nodes
nod = f'<nodes>\n    <node id="start" x="{x_min:.1f}" y="0.0"/>\n    <node id="end" x="{x_max:.1f}" y="0.0"/>\n</nodes>'
with open(f"{OUT_DIR}/highway.nod.xml","w") as f: f.write(nod)
print("Written: highway.nod.xml")

# Write edges — auto lanes from data
edg  = '<edges>\n'
edg += f'    <edge id="highway_right" from="start" to="end" numLanes="{right_lanes}" speed="{speed_ms:.2f}" spreadType="center"/>\n'
edg += f'    <edge id="highway_left"  from="end" to="start" numLanes="{left_lanes}"  speed="{speed_ms:.2f}" spreadType="center"/>\n'
edg += '</edges>'
with open(f"{OUT_DIR}/highway.edg.xml","w") as f: f.write(edg)
print("Written: highway.edg.xml")

# Run netconvert
result = subprocess.run([
    "netconvert",
    "--node-files", f"{OUT_DIR}/highway.nod.xml",
    "--edge-files", f"{OUT_DIR}/highway.edg.xml",
    "--output-file", f"{OUT_DIR}/highway.net.xml"
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
else:
    print("Written: highway.net.xml")

# Write routes
first_seen = df.groupby("NO")["SIMSEC"].min().to_dict()
veh_types  = df.groupby("NO")["VEHTYPE"].first().to_dict()
veh_links  = df.groupby("NO")["LANE_LINK_NO"].first().to_dict()
type_map   = {100:"car", 200:"truck", 300:"bus"}

rou  = '<routes>\n'
rou += '    <vType id="car"   accel="2.6" decel="4.5" sigma="0.5" length="4.5"  maxSpeed="50" color="yellow"/>\n'
rou += '    <vType id="truck" accel="1.3" decel="4.0" sigma="0.5" length="12.0" maxSpeed="36" color="red"/>\n'
rou += '    <vType id="bus"   accel="1.2" decel="4.0" sigma="0.5" length="15.0" maxSpeed="30" color="blue"/>\n'
rou += '    <route id="r_right" edges="highway_right"/>\n'
rou += '    <route id="r_left"  edges="highway_left"/>\n'

for veh_id, depart_time in sorted(first_seen.items(), key=lambda x: x[1]):
    vtype    = type_map.get(int(veh_types.get(veh_id, 100)), "car")
    veh_link = veh_links.get(veh_id, right_links[0] if right_links else links[0])
    route_id = "r_right" if veh_link in right_links else "r_left"
    lane_index = int(df[df["NO"]==veh_id]["LANE_INDEX"].iloc[0]) - 1
    rou += f'    <vehicle id="{veh_id}" type="{vtype}" route="{route_id}" depart="{depart_time}" departLane="{lane_index}"/>\n'
rou += '</routes>'
with open(f"{OUT_DIR}/highway.rou.xml","w") as f: f.write(rou)
print("Written: highway.rou.xml")
# Write SUMO config
cfg  = '<configuration>\n'
cfg += '    <input>\n'
cfg += '        <net-file value="highway.net.xml"/>\n'
cfg += '        <route-files value="highway.rou.xml"/>\n'
cfg += '    </input>\n'
cfg += '    <time>\n'
cfg += f'        <begin value="0"/>\n'
cfg += f'        <end value="{int(duration)+10}"/>\n'
cfg += '        <step-length value="0.1"/>\n'
cfg += '    </time>\n'
cfg += '    <report>\n'
cfg += '        <verbose value="true"/>\n'
cfg += '        <no-step-log value="true"/>\n'
cfg += '    </report>\n'
cfg += '</configuration>\n'
with open(f"{OUT_DIR}/highway.sumocfg","w") as f: f.write(cfg)
print("Written: highway.sumocfg")
print(f"\nDone. Network auto-built from CSV data.")
print(f"For a new video — just point CSV_PATH to new trajectories_xy.csv and run this script.")
