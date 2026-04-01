# with open("outputs/vissim/highway_network.inpx", encoding="utf-8") as f:
#     content = f.read()

# print(content[:3000])
# print(f"\nTotal file size: {len(content)} characters")

import cv2

cap = cv2.VideoCapture('data/videos/highway.mp4')
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'Width      : {w}')
print(f'Height     : {h}')
print(f'Orientation: {"Portrait" if h > w else "Landscape"}')
cap.release()