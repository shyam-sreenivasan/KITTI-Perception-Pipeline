"""
Complete Multi-Sensor Data Quality Analysis Pipeline

Runs end-to-end analysis:
1. Timestamp synchronization analysis
2. LiDAR data quality analysis
3. Camera data quality analysis
4. GPS/IMU data quality analysis
5. Calibration loading
6. Sensor fusion and projection validation

Outputs:
- JSON metrics for all sensors
- Visualizations for each analysis
- Fusion quality validation
- Comprehensive summary report
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt

# Import all modules
from perception.src.sensor_data_analyzers.KITTITimestampAnalyzer import KITTITimestampAnalyzer
from perception.src.sensor_data_analyzers.LidarDataAnalyzer import LiDARDataAnalyzer
from perception.src.sensor_data_analyzers.CameraDataAnalyzer import CameraDataAnalyzer
from perception.src.sensor_data_analyzers.GPSIMUDataAnalyzer import GPSIMUDataAnalyzer

from perception.src.data_loaders.KITTILiDARLoader import KITTILiDARLoader
from perception.src.data_loaders.KITTICameraLoader import KITTICameraLoader
from perception.src.data_loaders.KITTIOXTSLoader import KITTIOXTSLoader

from perception.src.calibration.CalibrationManager import CalibrationManager
from perception.src.SensorFusion import SensorFusion


def print_section_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def print_subsection(title):
    """Print formatted subsection."""
    print(f"\n### {title} ###")
    print("-"*70)


def run_complete_analysis(base_path: str, drive: str, output_dir: str = "outputs"):
    """
    Run complete multi-sensor data quality analysis pipeline.
    
    Args:
        base_path: Path to KITTI date directory (e.g., 'raw_data_downloader/2011_09_26')
        drive: Drive folder name (e.g., '2011_09_26_drive_0001_sync')
        output_dir: Directory for output files
    """
    base_path = Path(base_path)
    drive_path = base_path / drive
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Summary results
    summary = {
        'dataset': {
            'base_path': str(base_path),
            'drive': drive
        },
        'analyses': {}
    }
    
    # =========================================================================
    # STAGE 1: TIMESTAMP SYNCHRONIZATION ANALYSIS
    # =========================================================================
    print_section_header("STAGE 1: TIMESTAMP SYNCHRONIZATION ANALYSIS")
    
    timestamp_analyzer = KITTITimestampAnalyzer(drive_path)
    timestamp_analyzer.load_all_timestamps()
    timestamp_analyzer.analyze_frame_rates()
    timestamp_analyzer.analyze_synchronization(reference='lidar')
    timestamp_analyzer.print_summary()
    
    # Export
    timestamp_analyzer.export_to_json(output_path / "timestamp_analysis.json")
    
    # Store in summary
    summary['analyses']['timestamp_sync'] = timestamp_analyzer.get_summary_dict()
    
    print("\n✓ Timestamp analysis complete")
    
    # =========================================================================
    # STAGE 2: LIDAR DATA QUALITY ANALYSIS
    # =========================================================================
    print_section_header("STAGE 2: LIDAR DATA QUALITY ANALYSIS")
    
    # Initialize
    lidar_analyzer = LiDARDataAnalyzer()
    lidar_loader = KITTILiDARLoader(drive_path)
    
    # Single frame analysis
    print_subsection("Analyzing Frame 0")
    lidar_points = lidar_loader.load_frame(0)
    print(f"Loaded {len(lidar_points):,} LiDAR points")
    
    lidar_metrics = lidar_analyzer.analyze_frame(lidar_points, frame_idx=0, timestamp=0.0)
    print(f"Point count: {lidar_metrics['point_count']:,}")
    print(f"Range: {lidar_metrics['range_stats']['mean']:.2f}m (mean)")
    print(f"Horizontal FOV: {lidar_metrics['horizontal_fov_deg']:.1f}°")
    print(f"Spatial resolution: {lidar_metrics['spatial_resolution_cm']:.2f}cm")
    print(f"Quality flags: {lidar_metrics['quality_flags']}")
    
    # Visualize
    fig_lidar = lidar_analyzer.visualize_frame(
        lidar_points, 
        frame_idx=0,
        save_path=output_path / "lidar_frame_0_viz.png"
    )
    plt.close(fig_lidar)
    
    # Export
    lidar_analyzer.export_analysis(output_path / "lidar_frame_0.json", lidar_metrics)
    
    # Sample analysis
    print_subsection("Sample Analysis (5 random frames)")
    lidar_points_list, sample_indices = lidar_loader.load_sequence(0, 108)
    
    # Select random 5
    import numpy as np
    np.random.seed(42)
    sample_idx = sorted(np.random.choice(len(sample_indices), 5, replace=False))
    sampled_points = [lidar_points_list[i] for i in sample_idx]
    sampled_frames = [sample_indices[i] for i in sample_idx]
    
    lidar_sample_metrics = lidar_analyzer.analyze_sequence(
        sampled_points, sampled_frames, [i * 0.1 for i in sampled_frames]
    )
    lidar_analyzer.export_analysis(output_path / "lidar_sample.json", lidar_sample_metrics)
    print(f"Analyzed frames: {sampled_frames}")
    print(f"Mean points: {lidar_sample_metrics['summary']['point_count_mean']:.0f}")
    
    summary['analyses']['lidar'] = {
        'frame_0': lidar_metrics,
        'sample': lidar_sample_metrics['summary']
    }
    
    print("\n✓ LiDAR analysis complete")
    
    # =========================================================================
    # STAGE 3: CAMERA DATA QUALITY ANALYSIS
    # =========================================================================
    print_section_header("STAGE 3: CAMERA DATA QUALITY ANALYSIS")
    
    # Initialize
    camera_analyzer = CameraDataAnalyzer()
    camera_loader = KITTICameraLoader(drive_path, camera_id='image_02')
    
    # Single frame analysis
    print_subsection(f"Analyzing {camera_loader.camera_name} - Frame 0")
    img = camera_loader.load_frame(0)
    print(f"Loaded image: {img.shape}")
    
    camera_metrics = camera_analyzer.analyze_frame(img, frame_idx=0, timestamp=0.0)
    print(f"Resolution: {camera_metrics['resolution']}")
    print(f"Brightness: {camera_metrics['brightness_stats']['mean']:.2f}")
    print(f"Sharpness: {camera_metrics['sharpness']:.2f}")
    print(f"ORB features: {camera_metrics['orb_keypoints']}")
    print(f"Quality flags: {camera_metrics['quality_flags']}")
    
    # Visualize (basic mode)
    fig_camera = camera_analyzer.visualize_frame(
        img, 
        frame_idx=0,
        mode='basic',
        save_path=output_path / "camera_frame_0_viz.png"
    )
    plt.close(fig_camera)
    
    # Visualize (comprehensive mode)
    fig_camera_comp = camera_analyzer.visualize_frame(
        img,
        frame_idx=0,
        mode='comprehensive',
        save_path=output_path / "camera_frame_0_comprehensive.png"
    )
    plt.close(fig_camera_comp)
    
    # Export
    camera_analyzer.export_analysis(output_path / "camera_frame_0.json", camera_metrics)
    
    # Sample analysis
    print_subsection("Sample Analysis (first 5 frames)")
    images, cam_indices = camera_loader.load_sequence(0, 5)
    
    camera_sample_metrics = camera_analyzer.analyze_sequence(
        images, cam_indices, [i * 0.1 for i in cam_indices]
    )
    camera_analyzer.export_analysis(output_path / "camera_sample.json", camera_sample_metrics)
    print(f"Analyzed frames: {cam_indices}")
    print(f"Mean sharpness: {camera_sample_metrics['summary']['sharpness_mean']:.2f}")
    
    summary['analyses']['camera'] = {
        'camera_id': camera_loader.camera_id,
        'camera_name': camera_loader.camera_name,
        'frame_0': camera_metrics,
        'sample': camera_sample_metrics['summary']
    }
    
    print("\n✓ Camera analysis complete")
    
    # =========================================================================
    # STAGE 4: GPS/IMU DATA QUALITY ANALYSIS
    # =========================================================================
    print_section_header("STAGE 4: GPS/IMU DATA QUALITY ANALYSIS")
    
    # Initialize
    gps_analyzer = GPSIMUDataAnalyzer()
    gps_loader = KITTIOXTSLoader(drive_path)
    
    # Sequence analysis (GPS/IMU needs trajectory, not single frame)
    print_subsection("Sequence Analysis (108 frames)")
    gps_data_seq, gps_indices = gps_loader.load_sequence(0, 108)
    timestamps = [i * 0.1 for i in range(len(gps_indices))]
    
    gps_metrics = gps_analyzer.analyze_sequence(gps_data_seq, gps_indices, timestamps)
    
    gps_summary = gps_metrics['summary']
    print(f"Frames: {gps_summary['num_frames']}")
    print(f"Duration: {gps_summary['duration_sec']:.1f} seconds")
    print(f"Distance: {gps_summary['total_distance_m']:.2f} m")
    print(f"Avg speed: {gps_summary['speed_stats']['mean_kmh']:.1f} km/h")
    print(f"Heading change: {gps_summary['heading_change_deg']:.1f}°")
    print(f"Quality flags: {gps_summary['quality_flags']}")
    
    # Visualize
    fig_gps = gps_analyzer.visualize_sequence(
        gps_metrics,
        save_path=output_path / "gps_imu_sequence_viz.png"
    )
    plt.close(fig_gps)
    
    # Export
    gps_analyzer.export_analysis(output_path / "gps_imu_sequence.json", gps_metrics)
    
    summary['analyses']['gps_imu'] = gps_summary
    
    print("\n✓ GPS/IMU analysis complete")
    
    # =========================================================================
    # STAGE 5: CALIBRATION LOADING
    # =========================================================================
    print_section_header("STAGE 5: CALIBRATION LOADING")
    
    calib_mgr = CalibrationManager(base_path)
    calib_mgr.load_calibration()
    calib_mgr.print_summary()
    
    # Export calibration
    calib_mgr.export_calibration(output_path / "calibration.json")
    
    summary['analyses']['calibration'] = {
        'loaded': True,
        'cameras': 4,
        'transforms': ['lidar_to_camera', 'imu_to_lidar']
    }
    
    print("\n✓ Calibration loading complete")
    
    # =========================================================================
    # STAGE 6: SENSOR FUSION & PROJECTION VALIDATION
    # =========================================================================
    print_section_header("STAGE 6: SENSOR FUSION & PROJECTION VALIDATION")
    
    # Initialize fusion
    fusion = SensorFusion(calib_mgr)
    
    # Single frame fusion
    print_subsection("Single Frame Fusion (Frame 0)")
    
    # Load data (reuse from earlier)
    lidar_points_0 = lidar_loader.load_frame(0)
    image_0 = camera_loader.load_frame(0)
    
    # Project LiDAR to camera
    points_2d, depths, indices = fusion.project_lidar_to_camera(
        lidar_points_0,
        camera_id='02',
        filter_bounds=True,
        image_shape=image_0.shape
    )
    
    print(f"✓ Projection complete:")
    print(f"  Total LiDAR points: {len(lidar_points_0):,}")
    print(f"  Points in image: {len(points_2d):,}")
    print(f"  Projection rate: {len(points_2d)/len(lidar_points_0)*100:.1f}%")
    
    # Analyze projection quality
    fusion_metrics = fusion.analyze_projection_quality(
        lidar_points_0, points_2d, depths, indices,
        image_0.shape, frame_idx=0, timestamp=0.0
    )
    
    print(f"\nProjection quality:")
    print(f"  Horizontal coverage: {fusion_metrics['coverage']['horizontal_pct']:.1f}%")
    print(f"  Vertical coverage: {fusion_metrics['coverage']['vertical_pct']:.1f}%")
    print(f"  Mean depth: {fusion_metrics['depth_stats']['mean']:.2f}m")
    print(f"  Quality flags: {fusion_metrics['quality_flags']}")
    
    # Visualize single frame fusion
    fig_fusion = fusion.visualize_fusion(
        image_0, points_2d, depths,
        frame_idx=0,
        depth_range=(0, 50),
        save_path=output_path / "fusion_frame_0.png"
    )
    plt.close(fig_fusion)
    
    # Export
    fusion.export_metrics(output_path / "fusion_quality.json", fusion_metrics)
    
    # Multi-frame fusion comparison
    print_subsection("Multi-Frame Fusion Comparison")
    
    frame_indices = [0, 50, 100]
    fusion_results = []
    
    for frame_idx in frame_indices:
        # Load data
        lidar = lidar_loader.load_frame(frame_idx)
        img = camera_loader.load_frame(frame_idx)
        
        # Project
        pts_2d, deps, idx = fusion.project_lidar_to_camera(
            lidar, camera_id='02', filter_bounds=True, image_shape=img.shape
        )
        
        fusion_results.append({
            'frame_idx': frame_idx,
            'image': img,
            'points_2d': pts_2d,
            'depths': deps
        })
        
        print(f"  Frame {frame_idx:3d}: {len(pts_2d):,} projected points")
    
    # Visualize multi-frame comparison
    fig_multiframe = fusion.visualize_multiframe(
        fusion_results,
        depth_range=(0, 50),
        save_path=output_path / "fusion_multiframe.png"
    )
    plt.close(fig_multiframe)
    
    # Temporal region tracking
    print_subsection("Temporal Region Tracking")
    
    # Define region of interest (road area in front of vehicle)
    region_box = (400, 200, 800, 350)  # (x_min, y_min, x_max, y_max)
    
    region_metrics = fusion.analyze_region_temporal(
        fusion_results,
        region_box=region_box,
        timestamps=[0.0, 5.0, 10.0]
    )
    
    print(f"  Region: {region_box}")
    print(f"  Point count variation: {region_metrics['aggregate']['point_count_variation_pct']:.1f}%")
    print(f"  Consistency flags: {region_metrics['consistency_flags']}")
    
    # Visualize temporal analysis
    fig_temporal = fusion.visualize_temporal(
        fusion_results,
        region_box=region_box,
        save_path=output_path / "fusion_temporal.png"
    )
    plt.close(fig_temporal)
    
    # Export region metrics
    fusion.export_metrics(output_path / "region_temporal.json", region_metrics)
    
    summary['analyses']['fusion'] = {
        'frame_0': fusion_metrics,
        'multiframe': {
            'frames': frame_indices,
            'region_tracking': region_metrics['consistency_flags']
        }
    }
    
    print("\n✓ Sensor fusion analysis complete")
    
    # =========================================================================
    # GENERATE SUMMARY REPORT
    # =========================================================================
    print_section_header("GENERATING SUMMARY REPORT")
    
    # Export comprehensive summary
    with open(output_path / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Summary report generated")
    
    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================
    print_section_header("ANALYSIS COMPLETE - SUMMARY")
    
    print("\n### FILES GENERATED ###")
    print("\nJSON Metrics:")
    print("  ✓ timestamp_analysis.json         - Temporal synchronization")
    print("  ✓ lidar_frame_0.json              - LiDAR single frame")
    print("  ✓ lidar_sample.json               - LiDAR sample analysis")
    print("  ✓ camera_frame_0.json             - Camera single frame")
    print("  ✓ camera_sample.json              - Camera sample analysis")
    print("  ✓ gps_imu_sequence.json           - GPS/IMU trajectory")
    print("  ✓ calibration.json                - Calibration parameters")
    print("  ✓ fusion_quality.json             - Projection quality")
    print("  ✓ region_temporal.json            - Region tracking")
    print("  ✓ analysis_summary.json           - Complete summary")
    
    print("\nVisualizations:")
    print("  ✓ lidar_frame_0_viz.png           - LiDAR point cloud")
    print("  ✓ camera_frame_0_viz.png          - Camera quality (basic)")
    print("  ✓ camera_frame_0_comprehensive.png - Camera features (detailed)")
    print("  ✓ gps_imu_sequence_viz.png        - GPS/IMU trajectory")
    print("  ✓ fusion_frame_0.png              - LiDAR-camera fusion")
    print("  ✓ fusion_multiframe.png           - Multi-frame comparison")
    print("  ✓ fusion_temporal.png             - Temporal analysis")
    
    print("\n### KEY METRICS ###")
    print(f"\nTemporal Sync:")
    print(f"  • Max jitter: {summary['analyses']['timestamp_sync']['synchronization']['max_jitter_ms']:.2f}ms")
    print(f"  • Frame rate: {summary['analyses']['timestamp_sync']['sensors']['lidar']['frame_rate_hz']:.2f}Hz")
    
    print(f"\nLiDAR Quality:")
    print(f"  • Points/frame: {summary['analyses']['lidar']['frame_0']['point_count']:,}")
    print(f"  • Horizontal FOV: {summary['analyses']['lidar']['frame_0']['horizontal_fov_deg']:.1f}°")
    print(f"  • Spatial resolution: {summary['analyses']['lidar']['frame_0']['spatial_resolution_cm']:.2f}cm")
    
    print(f"\nCamera Quality:")
    print(f"  • Resolution: {summary['analyses']['camera']['frame_0']['resolution']}")
    print(f"  • Sharpness: {summary['analyses']['camera']['frame_0']['sharpness']:.2f}")
    print(f"  • ORB features: {summary['analyses']['camera']['frame_0']['orb_keypoints']}")
    
    print(f"\nGPS/IMU:")
    print(f"  • Distance: {summary['analyses']['gps_imu']['total_distance_m']:.2f}m")
    print(f"  • Avg speed: {summary['analyses']['gps_imu']['speed_stats']['mean_kmh']:.1f}km/h")
    print(f"  • Duration: {summary['analyses']['gps_imu']['duration_sec']:.1f}s")
    
    print(f"\nFusion Quality:")
    print(f"  • Projection rate: {summary['analyses']['fusion']['frame_0']['projection_rate_image_pct']:.1f}%")
    print(f"  • Horizontal coverage: {summary['analyses']['fusion']['frame_0']['coverage']['horizontal_pct']:.1f}%")
    print(f"  • Vertical coverage: {summary['analyses']['fusion']['frame_0']['coverage']['vertical_pct']:.1f}%")
    
    print("\n" + "="*70)
    print("✓ ALL ANALYSES COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_path.absolute()}/")
    
    return summary


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    BASE_PATH = "raw_data_downloader/2011_09_26"
    DRIVE = "2011_09_26_drive_0001_sync"
    OUTPUT_DIR = "outputs"
    
    # Run complete analysis
    summary = run_complete_analysis(BASE_PATH, DRIVE, OUTPUT_DIR)