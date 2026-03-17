================================================================================
  TRAFFIC VIDEO TO VISSIM SIMULATION PIPELINE
  README & USER GUIDE
================================================================================

WHAT THIS PROJECT DOES
------------------------
This pipeline takes any highway traffic video as input and automatically:
  1. Detects and tracks all vehicles (cars, trucks, buses) using YOLOv8
  2. Auto-detects lane count from white lane markings in the video
  3. Places lane boundaries using camera calibration (homography)
  4. Extracts traffic parameters per lane (flow, speed, headway, composition)
  5. Generates a VISSIM-ready .inpx simulation file
  6. Creates both carriageways (right from video, left mirrored)

================================================================================
  IF YOU HAVE A NEW VIDEO — WHAT TO CHANGE
================================================================================

  ONLY 3 THINGS NEED TO CHANGE:

  1. VIDEO FILE
     Copy your video to: data/videos/
     Update in configs/highway.yaml:
       video.path: "data/videos/YOUR_VIDEO.mp4"

  2. ROAD WIDTH (if different from current 11.0m)
     Update in configs/highway.yaml:
       lanes.total_width_m: 11.0
     Guide:
       2-lane road  →  7.0m
       3-lane road  → 11.0m  (current — UK motorway standard)
       4-lane road  → 14.0m
       5-lane road  → 17.5m

  3. REFERENCE POINTS (always required — camera position changes)
     Run: python scripts/phase1_preprocess.py
     Click 4 points on the road when window opens (~5 minutes)
     Press Q when done — config auto-updates

  LANE COUNT IS FULLY AUTOMATIC — no need to set it manually.
  The pipeline detects lane count from white dashed markings in the video.

================================================================================
  HOW AUTO LANE DETECTION WORKS
================================================================================
  Phase 3 detects lanes in two steps. First it counts the white dashed lane markings by
  scanning a horizontal strip of the frame and finding brightness peaks — each peak is 
  a white line, so N peaks means N+1 lanes. Then it divides the known road width 
  equally by that count and uses the homography matrix to project those division 
  points back onto the image as lane boundary lines. Vehicles play no role in this — it is 
  purely geometry and road marking detection.

  
  Phase 3 automatically counts lanes without any manual input:

  Step 1: Scan a horizontal strip at 75% frame height
  Step 2: Find brightness peaks = white lane marking lines
          (only within calibrated carriageway pixel range)
  Step 3: Lane count = peaks + 1
          (2 white lines between lanes = 3 lanes)
  Step 4: Clamp to realistic range (2-5 lanes)

  Example for highway.mp4:
    Carriageway range : pixel x=762 to x=1179
    Peaks found       : 2  (2 dashed lines)
    Lanes detected    : 3  ✓

  If auto-detection fails (worn markings), override in config:
    lanes.count: 3   (replace "auto" with actual number)

================================================================================
  RUN ORDER (always from project root folder)
================================================================================

  cd C:\Users\...\traffic_pipeline

  python scripts/phase1_preprocess.py   ← inspect video + pick 4 points
  python scripts/phase2_detect.py       ← YOLO detection + tracking
  python scripts/phase3_lanes.py        ← auto lane detection
  python scripts/phase4_extract.py      ← extract traffic parameters
  python scripts/phase5_vissim.py       ← generate VISSIM .inpx file

  Output: outputs/vissim/highway_network.inpx  → open in PTV VISSIM

================================================================================
  WHAT IS FULLY AUTOMATIC
================================================================================

  ✓ Lane count          auto-detected from white markings in video
  ✓ Lane widths         computed from total_width_m / lane count
  ✓ Lane positions      computed via homography matrix
  ✓ Video properties    FPS, resolution, frame count
  ✓ Speed distributions fitted from actual vehicle trajectory data
  ✓ Flow rates          counted from vehicle detections per lane
  ✓ Vehicle composition car/truck/bus % from YOLOv8 classification
  ✓ Left carriageway    geometry + parameters mirrored from right
  ✓ VISSIM XML          generated for any lane count dynamically

================================================================================
  WHAT YOU PROVIDE PER VIDEO (~10 minutes total)
================================================================================

  1. Video file              → copy to data/videos/
  2. total_width_m           → one number in config.yaml
  3. 4 reference point clicks → click on road markings in phase1

================================================================================
  PIPELINE DIAGRAM
================================================================================

  highway.mp4
      |
      v
  [Phase 1] Video inspection + homography calibration
      |       configs/highway.yaml updated automatically
      v
  [Phase 2] YOLOv8 + ByteTrack
      |       outputs/detection/trajectories.csv
      |       outputs/detection/detected.mp4
      v
  [Phase 3] Auto lane count + boundary placement
      |       outputs/lanes/lanes.json
      |       outputs/lanes/lanes_detected.jpg
      v
  [Phase 4] Traffic parameter extraction
      |       flow, speed, headway, composition per lane
      |       outputs/parameters/vissim_params.csv
      v
  [Phase 5] VISSIM .inpx generation
      |       6 links (3 right + 3 left mirrored)
      v
  outputs/vissim/highway_network.inpx
      |
      v
  Open in PTV VISSIM → validate → export to SUMO
      |
      v
  OMNeT++ + Veins → V2X communication simulation

================================================================================
  RESULTS FOR highway.mp4
================================================================================

  Video        : 10 seconds, 300 frames, 30 FPS
  Vehicles     : 40 unique (32 cars, 9 trucks, 5 buses)
  Lanes        : 3 auto-detected per carriageway

  Parameters extracted:
  Lane                     Flow      Speed    Headway  Cars  Trucks  Buses
  right_carriageway_lane1  1440 vph  46.1km/h  0.93s   80%   20%     0%
  right_carriageway_lane2  2520 vph  41.5km/h  1.60s  100%    0%     0%
  right_carriageway_lane3  2880 vph  40.3km/h  0.69s   30%   30%    40%

  Left carriageway         mirrored from right (same parameters)

  VISSIM file  : outputs/vissim/highway_network.inpx
  Links        : 6 total (3 right + 3 left)

================================================================================
  NEW VIDEO CHECKLIST
================================================================================

  [ ] Copy video to data/videos/
  [ ] Update video.path in configs/highway.yaml
  [ ] Update lanes.total_width_m if road width differs
  [ ] python scripts/phase1_preprocess.py  → click 4 road points → Q
  [ ] python scripts/phase2_detect.py
  [ ] python scripts/phase3_lanes.py
  [ ] Check outputs/lanes/lanes_detected.jpg (verify lanes look correct)
  [ ] python scripts/phase4_extract.py
  [ ] python scripts/phase5_vissim.py
  [ ] Open outputs/vissim/highway_network.inpx in VISSIM

================================================================================
  DEPENDENCIES
================================================================================

  pip install -r requirements.txt
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

  ultralytics, opencv-python, numpy, pandas, PyYAML,
  scipy, supervision, torch, torchvision

================================================================================
  PROJECT INFO
================================================================================

  Project  : Traffic Video to VISSIM Simulation Pipeline
  Phases   : 1-5 complete (video → VISSIM .inpx)
  Next     : SUMO export → OMNeT++ → V2X communication simulation

================================================================================
