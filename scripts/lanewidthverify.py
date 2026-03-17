import yaml

cfg = yaml.safe_load(open('configs/highway.yaml', encoding='utf-8'))
pts = cfg['calibration']['reference_points']
p1  = pts[0]['real']
p2  = pts[1]['real']

width = abs(p2[0] - p1[0])

print(f'P1 real: {p1}')
print(f'P2 real: {p2}')
print(f'Auto-computed width: {width}m')
print(f'Current config width: {cfg["lanes"]["total_width_m"]}m')
print(f'Match: {abs(width - cfg["lanes"]["total_width_m"]) < 0.01}')