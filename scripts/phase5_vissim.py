import yaml
import json
import pandas as pd
from pathlib import Path
import re

CONFIG_PATH = "configs/highway.yaml"
OUTPUT_DIR  = Path("outputs/vissim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_link_geometry(lanes_data, cfg):
    barrier_w = cfg["vissim"]["central_barrier_width_m"]
    links     = []
    link_no   = 1

    for lane in lanes_data:
        lane_w = lane["width_m"]
        idx    = lane["lane_index"]
        x_mid  = round((idx - 1) * lane_w + lane_w / 2, 3)
        links.append({
            "no":         link_no,
            "name":       lane["lane_id"],
            "side":       "right",
            "lane_width": lane_w,
            "x1": x_mid,  "y1": 0.0,
            "x2": x_mid,  "y2": 200.0,
        })
        link_no += 1

    if cfg["vissim"].get("create_both_carriageways", False):
        for lane in lanes_data:
            lane_w = lane["width_m"]
            idx    = lane["lane_index"]
            x_mid  = round(-((idx - 1) * lane_w + lane_w / 2) - barrier_w, 3)
            links.append({
                "no":         link_no,
                "name":       lane["lane_id"].replace(
                              "right_carriageway", "left_carriageway"),
                "side":       "left",
                "lane_width": lane_w,
                "x1": x_mid,  "y1": 200.0,
                "x2": x_mid,  "y2": 0.0,
            })
            link_no += 1

    return links

def build_links_xml(links):
    xml = "\t<links>\n"
    for lk in links:
        xml += (
            f'\t\t<link no="{lk["no"]}" name="{lk["name"]}" '
            f'anmFlag="false" displayType="1" '
            f'linkBehavType="3" level="1">\n'
            f'\t\t\t<geometry>\n'
            f'\t\t\t\t<points3D>\n'
            f'\t\t\t\t\t<point3D x="{lk["x1"]}" '
            f'y="{lk["y1"]}" zOffset="0"/>\n'
            f'\t\t\t\t\t<point3D x="{lk["x2"]}" '
            f'y="{lk["y2"]}" zOffset="0"/>\n'
            f'\t\t\t\t</points3D>\n'
            f'\t\t\t</geometry>\n'
            f'\t\t\t<lanes>\n'
            f'\t\t\t\t<lane no="1" width="{lk["lane_width"]}"/>\n'
            f'\t\t\t</lanes>\n'
            f'\t\t</link>\n'
        )
    xml += "\t</links>\n"
    return xml

def build_speed_distributions_xml(vissim_df):
    xml     = ""
    spd_map = {}
    spd_no  = 2001

    for _, row in vissim_df.iterrows():
        mean = float(row["mean_speed_kmh"]) if row["mean_speed_kmh"] > 0 else 80.0
        std  = float(row["std_speed_kmh"])  if row["std_speed_kmh"]  > 0 else 10.0
        lid  = row["lane_id"]

        xml += (
            f'\t\t<speedDistribution no="{spd_no}" '
            f'name="SpeedDist_{lid}">\n'
            f'\t\t\t<speedDistrDatPts>\n'
        )
        for spd_val, cdf in [
            (max(20.0, mean - 2*std), 0.0),
            (max(20.0, mean - std),   0.15),
            (mean,                    0.5),
            (mean + std,              0.85),
            (mean + 2*std,            1.0),
        ]:
            xml += (
                f'\t\t\t\t<speedDistributionDataPoint '
                f'x="{round(spd_val, 1)}" fx="{cdf}"/>\n'
            )
        xml += "\t\t\t</speedDistrDatPts>\n\t\t</speedDistribution>\n"
        spd_map[lid] = spd_no
        spd_no += 1

    # Mirror for left carriageway
    for _, row in vissim_df.iterrows():
        left_id = row["lane_id"].replace(
                  "right_carriageway", "left_carriageway")
        spd_map[left_id] = spd_map[row["lane_id"]]

    return xml, spd_map

def build_compositions_xml(vissim_df, spd_map):
    xml      = ""
    comp_map = {}
    comp_no  = 101

    all_ids = list(vissim_df["lane_id"]) + [
        lid.replace("right_carriageway", "left_carriageway")
        for lid in vissim_df["lane_id"]
    ]

    for lane_id in all_ids:
        right_id = lane_id.replace("left_carriageway", "right_carriageway")
        row      = vissim_df[vissim_df["lane_id"] == right_id]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        pct_car   = float(row.get("pct_car",   0.8))
        pct_truck = float(row.get("pct_truck", 0.15))
        pct_bus   = float(row.get("pct_bus",   0.05))
        total     = pct_car + pct_truck + pct_bus
        if total == 0:
            pct_car = 1.0; total = 1.0
        pct_car   = round(pct_car   / total, 3)
        pct_truck = round(pct_truck / total, 3)
        pct_bus   = round(1.0 - pct_car - pct_truck, 3)

        sd = spd_map.get(lane_id, 2001)

        xml += (
            f'\t\t<vehicleComposition no="{comp_no}" '
            f'name="Comp_{lane_id}">\n'
            f'\t\t\t<vehCompRelFlows>\n'
        )
        if pct_car > 0:
            xml += (
                f'\t\t\t\t<vehicleCompositionRelativeFlow '
                f'vehType="100" desSpeedDistr="{sd}" '
                f'relFlow="{pct_car}"/>\n'
            )
        if pct_truck > 0:
            xml += (
                f'\t\t\t\t<vehicleCompositionRelativeFlow '
                f'vehType="200" desSpeedDistr="{sd}" '
                f'relFlow="{pct_truck}"/>\n'
            )
        if pct_bus > 0:
            xml += (
                f'\t\t\t\t<vehicleCompositionRelativeFlow '
                f'vehType="300" desSpeedDistr="{sd}" '
                f'relFlow="{pct_bus}"/>\n'
            )
        xml += "\t\t\t</vehCompRelFlows>\n\t\t</vehicleComposition>\n"
        comp_map[lane_id] = comp_no
        comp_no += 1

    return xml, comp_map

def build_inputs_xml(links, vissim_df, comp_map):
    xml = "\t<vehicleInputs>\n"

    for lk in links:
        lane_id  = lk["name"]
        flow_row = vissim_df[vissim_df["lane_id"] == lane_id]
        if len(flow_row) > 0:
            flow_vph = int(flow_row.iloc[0]["flow_vph"])
        else:
            right_id = lane_id.replace("left_carriageway", "right_carriageway")
            r_row    = vissim_df[vissim_df["lane_id"] == right_id]
            flow_vph = int(r_row.iloc[0]["flow_vph"]) if len(r_row) > 0 else 1000

        comp   = comp_map.get(lane_id, 101)
        inp_no = lk["no"]

        xml += (
            f'\t\t<vehicleInput anmFlag="false" '
            f'link="{lk["no"]}" name="{lane_id}" '
            f'no="{inp_no}">\n'
            f'\t\t\t<timeIntVehVols>\n'
            f'\t\t\t\t<timeIntervalVehVolume cont="false" '
            f'timeInt="1 0" vehComp="{comp}" '
            f'volType="STOCHASTIC" volume="{flow_vph}"/>\n'
            f'\t\t\t</timeIntVehVols>\n'
            f'\t\t</vehicleInput>\n'
        )

    xml += "\t</vehicleInputs>\n"
    return xml

def inject_into_template(template, links_xml, spd_xml,
                          comp_xml, inputs_xml):
    output = template

    # Remove any existing links and vehicleInputs from template
    # (in case template had test data added manually)
    output = re.sub(
        r'\t<links>.*?</links>\n',
        '', output, flags=re.DOTALL
    )
    output = re.sub(
        r'\t<vehicleInputs>.*?</vehicleInputs>\n',
        '', output, flags=re.DOTALL
    )

    # Inject speed distributions inside existing block
    output = output.replace(
        "\t</speedDistributions>",
        spd_xml + "\t</speedDistributions>"
    )

    # Inject compositions inside existing block
    output = output.replace(
        "\t</vehicleCompositions>",
        comp_xml + "\t</vehicleCompositions>"
    )

    # Inject links and inputs before closing tag
    output = output.replace(
        "</network>",
        "\n" + links_xml + "\n" + inputs_xml + "</network>"
    )

    return output

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Phase 5: VISSIM 2026 .inpx Generation")
    print("="*50)

    cfg       = load_config(CONFIG_PATH)
    lanes     = json.load(open("outputs/lanes/lanes.json"))
    vissim_df = pd.read_csv("outputs/parameters/vissim_params.csv")

    template_path = "configs/vissim_template.inpx"
    if not Path(template_path).exists():
        print(f"ERROR: Template not found at {template_path}")
        print("Save an empty VISSIM network as configs/vissim_template.inpx")
        exit(1)

    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    print("\n▶ Computing link geometry...")
    links = compute_link_geometry(lanes, cfg)
    print(f"  Total links: {len(links)}")
    for lk in links:
        print(f"    Link {lk['no']:2d}: {lk['name']}")

    print("\n▶ Building speed distributions...")
    spd_xml, spd_map = build_speed_distributions_xml(vissim_df)
    print(f"  Speed distributions: {len(spd_map)}")

    print("\n▶ Building vehicle compositions...")
    comp_xml, comp_map = build_compositions_xml(vissim_df, spd_map)
    print(f"  Compositions: {len(comp_map)}")

    print("\n▶ Building vehicle inputs...")
    inputs_xml = build_inputs_xml(links, vissim_df, comp_map)

    print("\n▶ Building links XML...")
    links_xml = build_links_xml(links)

    print("\n▶ Injecting into VISSIM 2026 template...")
    output = inject_into_template(
        template, links_xml, spd_xml, comp_xml, inputs_xml
    )

    out_path = OUTPUT_DIR / "highway_network.inpx"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n  ✓ Saved: {out_path}")
    print("\n  Network summary:")
    print("-"*50)
    for lk in links:
        lane_id  = lk["name"]
        flow_row = vissim_df[vissim_df["lane_id"] == lane_id]
        if len(flow_row) > 0:
            flow = int(flow_row.iloc[0]["flow_vph"])
            spd  = flow_row.iloc[0]["mean_speed_kmh"]
        else:
            right_id = lane_id.replace(
                       "left_carriageway", "right_carriageway")
            r_row    = vissim_df[vissim_df["lane_id"] == right_id]
            flow = int(r_row.iloc[0]["flow_vph"])  if len(r_row) > 0 else 0
            spd  = r_row.iloc[0]["mean_speed_kmh"] if len(r_row) > 0 else 0
        print(f"    {lane_id:40s} "
              f"flow={flow:5d} vph  speed={spd:.1f} km/h")

    print("\n" + "="*50)
    print("  Phase 5 Complete!")
    print(f"  VISSIM file: {out_path}")
    print("="*50)