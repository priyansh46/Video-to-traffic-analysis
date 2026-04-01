# scripts/parse_inpx_geometry.py
import xml.etree.ElementTree as ET
import json

def extract_link_geometry(inpx_path):
    tree = ET.parse(inpx_path)
    root = tree.getroot()
    
    links = {}
    for link in root.findall('.//link'):
        link_no   = int(link.get('no'))
        pts = link.findall('.//linkPolyPts/linkPolyPoint')
        
        # Get start and end points
        if len(pts) >= 2:
            x1 = float(pts[0].get('x'))
            y1 = float(pts[0].get('y'))
            x2 = float(pts[-1].get('x'))
            y2 = float(pts[-1].get('y'))
            
            import math
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle  = math.atan2(y2-y1, x2-x1)
            
            links[link_no] = {
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'length': length,
                'angle':  angle
            }
    
    return links

if __name__ == "__main__":
    links = extract_link_geometry(
        "outputs/vissim/highway_network.inpx"
    )
    with open("outputs/vissim/link_geometry.json", "w") as f:
        json.dump(links, f, indent=2)
    print(f"Extracted {len(links)} links")
    for k, v in links.items():
        print(f"  Link {k}: ({v['x1']:.1f},{v['y1']:.1f}) → "
              f"({v['x2']:.1f},{v['y2']:.1f}), "
              f"length={v['length']:.1f}m")