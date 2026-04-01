import pandas as pd

def convert_to_ns2_trace(input_csv, output_txt):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    with open(output_txt, 'w') as f:
        # Sort by vehicle ID, then time
        df = df.sort_values(by=['NO', 'SIMSEC'])
        
        for vid in df['NO'].unique():
            veh_data = df[df['NO'] == vid]
            
            # Initial position
            first_row = veh_data.iloc[0]
            f.write(f"$node_({int(vid)}) set X_ {first_row['X']:.2f}\n")
            f.write(f"$node_({int(vid)}) set Y_ {first_row['Y']:.2f}\n")
            f.write(f"$node_({int(vid)}) set Z_ 0.0\n")
            
            # Waypoints
            for _, row in veh_data.iterrows():
                speed_ms = row['SPEED'] / 3.6  # convert km/h to m/s
                # NS2 format: $ns_ at <time> "$node_(<id>) setdest <x> <y> <speed>"
                f.write(f"$ns_ at {row['SIMSEC']:.2f} \"$node_({int(vid)}) setdest {row['X']:.2f} {row['Y']:.2f} {speed_ms:.2f}\"\n")

    print(f"Success! NS2 trace saved to {output_txt}")

if __name__ == "__main__":
    convert_to_ns2_trace("outputs/vissim/trajectories_xy.csv", "outputs/vissim/omnet_mobility.txt")