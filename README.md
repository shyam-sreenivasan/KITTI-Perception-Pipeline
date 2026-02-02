# End-to-End Autonomous Vehicle Perception Pipeline

**Production-grade multi-sensor perception system for autonomous driving**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![KITTI Dataset](https://img.shields.io/badge/dataset-KITTI-green.svg)](http://www.cvlibs.net/datasets/kitti/)
---

## 🎯 Project Goal

Build a complete autonomous vehicle perception pipeline: **Raw Sensors → World Model → Planner-Ready JSON**

Transform synchronized multi-sensor data (Camera, LiDAR, GPS/IMU) into a structured world representation with tracked objects, occupancy grid, and ego pose for downstream motion planning.

---

## 🏗️ Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              RAW SENSOR DATA                                │
│  • Camera (4× RGB/Grayscale @ 10Hz)                         │
│  • LiDAR (Velodyne HDL-64E @ 10Hz, 120K pts/frame)         │
│  • GPS/IMU (OXTS @ 10Hz, 30-value packets)                 │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 1-2: PREPROCESSING & SYNCHRONIZATION ✅ DONE      │
│  • Temporal alignment validation (3ms jitter)               │
│  • Frame-level sensor synchronization                       │
│  • Data completeness verification                           │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 3-4: SENSOR DATA QUALITY ANALYSIS ✅ DONE         │
│  • LiDAR: Coverage, density, resolution analysis            │
│  • Camera: Sharpness, exposure, feature detection           │
│  • GPS/IMU: Trajectory, velocity, orientation               │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 5-6: CALIBRATION & SENSOR FUSION ✅ DONE          │
│  • Intrinsic/extrinsic parameter loading                    │
│  • LiDAR-to-camera projection (85% visibility)              │
│  • Multi-frame alignment validation (<5px error)            │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 7: OBJECT DETECTION 🚧 ONGOING                    │
│  • 2D detection (YOLOv8 on camera images)                   │
│  • Bounding box extraction [x1, y1, x2, y2, class, conf]   │
│  • Confidence filtering & NMS                               │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 8: 3D OBJECT LOCALIZATION 🚧 ONGOING              │
│  • LiDAR-2D bbox association (point-in-box)                │
│  • 3D position estimation (median/centroid)                 │
│  • Coordinate transform to vehicle frame                    │
│  • Output: Object3D(id, class, x, y, z)                    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 9: MULTI-OBJECT TRACKING 🚧 ONGOING               │
│  • Kalman filter (state: x, y, vx, vy)                     │
│  • Data association (nearest-neighbor in BEV)               │
│  • Track lifecycle management (birth/death)                 │
│  • Ego-motion compensation                                  │
│  • Output: Tracked objects with IDs & velocities           │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 10: OCCUPANCY GRID GENERATION 📋 PLANNED          │
│  • BEV grid creation (50m × 50m @ 0.5m resolution)         │
│  • LiDAR point → occupied cell mapping                      │
│  • Free space inference                                     │
│  • Grid alignment with ego pose                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│     STAGE 11: WORLD MODEL & PLANNER OUTPUT 📋 PLANNED       │
│  • Aggregate: ego_pose + tracked_objects + occupancy_grid  │
│  • JSON serialization for planner interface                 │
│  • Temporal consistency validation                          │
│  • 10Hz world model stream                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Completed Work (Stages 1-6)

### **Stage 1-2: Data Pipeline & Synchronization**
- ✅ Multi-sensor data loader (Camera, LiDAR, GPS/IMU @ 10Hz)
- ✅ Temporal synchronization analysis (3ms max jitter)
- ✅ Frame completeness verification (0% missing)
- ✅ Iterator-based pipeline (memory-efficient)

### **Stage 3-4: Sensor Data Quality Analysis**
- ✅ **LiDAR Analysis:**
  - 120,453 points/frame (1% variation)
  - 360° horizontal coverage, 26.9° vertical FOV
  - 2.6cm spatial resolution @ 5m
  - 78% non-zero reflectance
  
- ✅ **Camera Analysis:**
  - 1,847 ORB keypoints/frame
  - 387.2 sharpness score (Laplacian variance)
  - 0.3% overexposed, 2.1% underexposed
  - 8.45% edge density
  
- ✅ **GPS/IMU Analysis:**
  - 78.4m trajectory over 10.8s
  - 26.1 km/h average speed
  - 45° heading change (left turn)
  - 8-12 GPS satellites (good fix)

### **Stage 5-6: Calibration & Sensor Fusion**
- ✅ Calibration loading (3 files: cam_to_cam, velo_to_cam, imu_to_velo)
- ✅ LiDAR-to-camera projection pipeline
- ✅ 85.3% point visibility in camera FOV
- ✅ 71.2% points within image bounds
- ✅ <5 pixel alignment error
- ✅ Multi-frame validation (frames 0, 50, 100)
- ✅ Temporal region tracking (road area stability)

---

## 🚧 Ongoing Work (Stages 7-9)

### **Stage 7: Object Detection**
**Status:** Implementation in progress

**Tasks:**
- [ ] YOLOv8 model integration
- [ ] Frame-by-frame detection
- [ ] Confidence thresholding
- [ ] Class filtering (vehicles, pedestrians, cyclists)
- [ ] Detection visualization

**Expected Output:** `[{bbox: [x1, y1, x2, y2], class: 'car', conf: 0.95}]`

### **Stage 8: 3D Object Localization**
**Status:** Design phase

**Tasks:**
- [ ] Point-in-box association (LiDAR ↔ 2D bbox)
- [ ] 3D centroid estimation
- [ ] Coordinate frame transformation
- [ ] 3D bounding box fitting
- [ ] Depth validation

**Expected Output:** `[{id: 1, class: 'car', position: (x, y, z), size: (l, w, h)}]`

### **Stage 9: Multi-Object Tracking**
**Status:** Design phase

**Tasks:**
- [ ] Kalman filter implementation (constant velocity model)
- [ ] Data association (nearest-neighbor / Hungarian)
- [ ] Track lifecycle (initialization, maintenance, termination)
- [ ] Ego-motion compensation (using GPS/IMU)
- [ ] Track smoothing

**Expected Output:** `[{track_id: 3, class: 'car', x, y, vx, vy, age: 15}]`

---

## 📋 Planned Work (Stages 10-11)

### **Stage 10: BEV Occupancy Grid**
- [ ] Grid initialization (50m × 50m, 0.5m cells)
- [ ] LiDAR point → cell mapping
- [ ] Free space inference
- [ ] Static obstacle marking
- [ ] Ego-centric alignment

### **Stage 11: World Model Output**
- [ ] JSON schema definition
- [ ] Ego pose serialization
- [ ] Tracked objects serialization
- [ ] Occupancy grid serialization
- [ ] 10Hz output stream

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download KITTI dataset
# Place in: raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync/

# Run complete analysis (Stages 1-6)
python main.py
```

**Outputs:** 10 JSON metrics + 7 visualizations in `outputs/`

---

## 📁 Project Structure

```
perception/
├── src/
│   ├── calibration/              # Calibration parameter management
│   │   └── CalibrationManager.py
│   │
│   ├── data_loaders/             # KITTI dataset loaders
│   │   ├── KITTILiDARLoader.py
│   │   ├── KITTICameraLoader.py
│   │   └── KITTIOXTSLoader.py
│   │
│   ├── sensor_data_analyzers/    # Data quality analysis
│   │   ├── KITTITimestampAnalyzer.py   # Temporal sync
│   │   ├── LidarDataAnalyzer.py        # Point cloud quality
│   │   ├── CameraDataAnalyzer.py       # Image quality
│   │   └── GPSIMUDataAnalyzer.py       # Trajectory quality
│   │
│   ├── SensorFusion.py           # Multi-sensor fusion & projection
│   └── utils.py                  # Shared utilities
│
├── main.py                       # Complete pipeline runner
├── outputs/                      # Generated metrics & visualizations
├── requirements.txt
└── README.md
```

---

## 🔬 Usage Example

```python
from perception.src.calibration.CalibrationManager import CalibrationManager
from perception.src.SensorFusion import SensorFusion
from perception.src.data_loaders.KITTILiDARLoader import KITTILiDARLoader
from perception.src.data_loaders.KITTICameraLoader import KITTICameraLoader

# Load calibration
calib = CalibrationManager('raw_data_downloader/2011_09_26')
calib.load_calibration()

# Initialize fusion
fusion = SensorFusion(calib)

# Load sensor data
lidar_loader = KITTILiDARLoader('path/to/drive')
camera_loader = KITTICameraLoader('path/to/drive', 'image_02')

lidar_points = lidar_loader.load_frame(0)
image = camera_loader.load_frame(0)

# Project LiDAR to camera
points_2d, depths, indices = fusion.project_lidar_to_camera(
    lidar_points, camera_id='02', filter_bounds=True, image_shape=image.shape
)

# Analyze quality
metrics = fusion.analyze_projection_quality(
    lidar_points, points_2d, depths, indices, image.shape
)

# Visualize
fig = fusion.visualize_fusion(image, points_2d, depths)
```

---

## 📊 Generated Outputs

### **JSON Metrics (10 files)**
- `timestamp_analysis.json` - Temporal synchronization metrics
- `lidar_frame_0.json` - LiDAR single frame quality
- `lidar_sample.json` - LiDAR sample statistics
- `camera_frame_0.json` - Camera image quality
- `camera_sample.json` - Camera sample statistics
- `gps_imu_sequence.json` - GPS/IMU trajectory analysis
- `calibration.json` - Calibration parameters
- `fusion_quality.json` - Projection validation metrics
- `region_temporal.json` - Region tracking over time
- `analysis_summary.json` - Complete validation report

### **Visualizations (7 files)**
- `lidar_frame_0_viz.png` - Point cloud (3D, BEV, range, azimuth)
- `camera_frame_0_viz.png` - Image quality (4-panel basic)
- `camera_frame_0_comprehensive.png` - Feature analysis (9-panel detailed)
- `gps_imu_sequence_viz.png` - Trajectory dashboard (6-panel)
- `fusion_frame_0.png` - LiDAR-camera overlay (depth-colored)
- `fusion_multiframe.png` - Multi-frame comparison (3 frames)
- `fusion_temporal.png` - Temporal tracking dashboard

---

## ⏭️ Roadmap

### **Phase 1: Data Foundation** ✅ COMPLETE (Stages 1-6)
- [x] Multi-sensor data loading
- [x] Synchronization validation  
- [x] Quality analysis (LiDAR, Camera, GPS/IMU)
- [x] Calibration & fusion

### **Phase 2: Core Perception** 🚧 IN PROGRESS (Stages 7-9)
- [ ] 2D object detection (YOLOv8)
- [ ] 3D bounding box estimation
- [ ] Multi-object tracking (Kalman filter)

### **Phase 3: World Model** 📋 PLANNED (Stages 10-11)
- [ ] BEV occupancy grid
- [ ] Planner-ready JSON output
- [ ] Temporal prediction

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Core Libraries:** NumPy, OpenCV, Matplotlib, SciPy, Pandas
- **Dataset:** KITTI Raw (Synced) - 2011_09_26_drive_0001
- **Sensors:** 
  - Velodyne HDL-64E LiDAR (64 beams, 360°)
  - Point Grey Flea2 cameras (1242×375, stereo)
  - OXTS RT3000 GPS/IMU (RTK-GPS)

---

## 📚 Documentation

- `outputs/analysis_summary.json` - Complete validation report
- Individual JSON files for each sensor analysis
- Inline code documentation (docstrings)

---

## 🙏 Acknowledgments

- **KITTI Dataset:** Karlsruhe Institute of Technology

---

## 📧 Contact

For questions or collaboration: shyamsreeni.612@gmail.com

---

**Project Status:** ✅ Data validation complete (6/11 stages) | 🚧 Object detection ongoing (Stage 7)
