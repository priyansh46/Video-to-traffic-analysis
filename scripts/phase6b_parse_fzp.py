# scripts/parse_fzp.py
import pandas as pd
import json
import math

def load_fzp(fzp_path):
    """Parse VISSIM vehicle record .fzp file"""
    rows = []
    with open(fzp_path, encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith('$VEHICLE:'):
                # Header line
                header = line.replace('$VEHICLE:', '').split(';')
            elif line.startswith('*') or line.startswith('$'):
                continue  # skip comment lines
            elif header and line:
                vals = line.split(';')
                if len(vals) == len(header):
                    rows.append(dict(zip(header, vals)))
    
    df = pd.DataFrame(rows)
    # Convert types
    df['SIMSEC']        = df['SIMSEC'].astype(float)
    df['NO']            = df['NO'].astype(float).astype(int)
    df['LANE\\LINK\\NO']= df['LANE\\LINK\\NO'].astype(float).astype(int)
    df['LANE\\INDEX']   = df['LANE\\INDEX'].astype(float).astype(int)
    df['POS']           = df['POS'].astype(float)
    df['POSLAT']        = df['POSLAT'].astype(float)
    df['SPEED']         = df['SPEED'].astype(float)  # km/h
    df['VEHTYPE']       = df['VEHTYPE'].astype(int)
    return df

def pos_to_xy(link_no, pos, poslat, links):
    """
    Convert VISSIM link position to absolute X, Y
    POS    = distance along link centre line (metres)
    POSLAT = lateral offset from centre (metres)
    """
    if link_no not in links:
        return None, None
    
    lk     = links[link_no]
    angle  = lk['angle']
    length = lk['length']
    
    # Clamp pos to link length
    pos = min(pos, length)
    
    # Point along centre line
    cx = lk['x1'] + pos * math.cos(angle)
    cy = lk['y1'] + pos * math.sin(angle)
    
    # Lateral offset (perpendicular to link direction)
    perp_angle = angle + math.pi / 2
    x = cx + poslat * math.cos(perp_angle)
    y = cy + poslat * math.sin(perp_angle)
    
    return round(x, 3), round(y, 3)

def convert_to_absolute(df, links):
    """Add absolute X, Y columns to dataframe"""
    xs, ys = [], []
    for _, row in df.iterrows():
        x, y = pos_to_xy(
            row['LANE\\LINK\\NO'],
            row['POS'],
            row['POSLAT'],
            links
        )
        xs.append(x)
        ys.append(y)
    df['X'] = xs
    df['Y'] = ys
    return df.dropna(subset=['X', 'Y'])

if __name__ == "__main__":
    with open("outputs/vissim/link_geometry.json") as f:
        links = json.load(f)
    # JSON keys are strings, convert to int
    links = {int(k): v for k, v in links.items()}
    
    df = load_fzp("outputs/vissim/highway_network.fzp")
    df = convert_to_absolute(df, links)
    
    df.to_csv("outputs/vissim/trajectories_xy.csv", index=False)
    print(f"Converted {len(df)} records")
    print(df[['SIMSEC','NO','X','Y','SPEED']].head(10))