"""
Multi-sensor fusion module for autonomous vehicle perception.

Provides:
- LiDAR-to-camera projection
- Projection quality validation
- Region-based analysis
- Temporal consistency tracking
- Multi-modal visualization

Requires CalibrationManager for coordinate transforms.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json


def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


@dataclass
class ProjectionQualityMetrics:
    """Metrics for LiDAR-to-camera projection quality."""
    frame_idx: int
    timestamp: float
    total_lidar_points: int
    points_in_camera_fov: int
    points_in_image_bounds: int
    projection_rate_fov_pct: float
    projection_rate_image_pct: float
    depth_stats: Dict[str, float]
    coverage: Dict[str, float]
    depth_distribution: Dict[str, int]
    region_analysis: Dict[str, Dict[str, float]]
    quality_flags: Dict[str, bool]


class SensorFusion:
    """
    Multi-sensor data fusion for perception systems.
    
    Handles coordinate transformations, projection, and quality validation
    for fusing LiDAR and camera data.
    
    Example:
        >>> from calibration import CalibrationManager
        >>> 
        >>> calib = CalibrationManager('data/2011_09_26')
        >>> calib.load_calibration()
        >>> 
        >>> fusion = SensorFusion(calib)
        >>> 
        >>> # Project LiDAR to camera
        >>> points_2d, depths, indices = fusion.project_lidar_to_camera(
        ...     lidar_points, camera_id='02', filter_bounds=True, image_shape=image.shape
        ... )
        >>> 
        >>> # Analyze quality
        >>> metrics = fusion.analyze_projection_quality(
        ...     lidar_points, points_2d, depths, indices, image.shape
        ... )
        >>> 
        >>> # Visualize
        >>> fig = fusion.visualize_fusion(image, points_2d, depths)
    """
    
    # Configuration
    DEPTH_BINS = [0, 10, 20, 30, 40, 50, 100]  # meters
    
    # Standard image regions (vertical thirds)
    IMAGE_REGIONS = {
        'top_sky': (0.0, 0.33),      # Top third (sky)
        'middle_objects': (0.33, 0.67),  # Middle third (objects)
        'bottom_road': (0.67, 1.0)    # Bottom third (road)
    }
    
    def __init__(self, calibration_manager):
        """
        Initialize sensor fusion module.
        
        Args:
            calibration_manager: CalibrationManager instance with loaded calibration
        """
        self.calib_mgr = calibration_manager
        
        if not self.calib_mgr.calibration:
            raise ValueError("CalibrationManager has no loaded calibration. "
                           "Call load_calibration() first.")
    
    # =========================================================================
    # PROJECTION METHODS
    # =========================================================================
    
    def project_lidar_to_camera(
        self,
        lidar_points: np.ndarray,
        camera_id: str = '02',
        filter_bounds: bool = True,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project LiDAR points to camera image coordinates.
        
        Transformation pipeline:
        1. Velodyne → Camera (unrectified): Apply R, T
        2. Camera → Rectified: Apply R_rect
        3. Filter points in front of camera (Z > 0)
        4. Rectified → Image: Apply P_rect projection
        5. Optionally filter points within image bounds
        
        Args:
            lidar_points: (N, 4) array [x, y, z, reflectance]
            camera_id: Target camera ('00', '01', '02', '03')
            filter_bounds: If True, remove points outside image
            image_shape: (height, width) for bounds filtering
            
        Returns:
            points_2d: (M, 2) pixel coordinates [u, v]
            depths: (M,) distances from camera (meters)
            valid_indices: (M,) original indices in lidar_points
        """
        # Get calibration parameters
        velo_to_cam = self.calib_mgr.get_lidar_to_camera_transform()
        cam_intrinsics = self.calib_mgr.get_camera_intrinsics(camera_id)
        
        R_velo = velo_to_cam['R']
        T_velo = velo_to_cam['T']
        R_rect = cam_intrinsics['R_rect']
        P_rect = cam_intrinsics['P_rect']
        
        # Extract 3D coordinates
        points_3d = lidar_points[:, :3]  # (N, 3)
        
        # Step 1: Velodyne → Camera coordinate system
        points_cam = (R_velo @ points_3d.T).T + T_velo.T
        
        # Step 2: Apply rectification
        points_cam_rect = (R_rect @ points_cam.T).T
        
        # Step 3: Filter points in front of camera (positive Z)
        valid_mask = points_cam_rect[:, 2] > 0
        points_cam_rect_valid = points_cam_rect[valid_mask]
        depths = points_cam_rect_valid[:, 2]
        valid_indices = np.where(valid_mask)[0]
        
        # Step 4: Project to image plane
        points_hom = np.hstack([points_cam_rect_valid,
                                np.ones((points_cam_rect_valid.shape[0], 1))])
        points_2d_hom = (P_rect @ points_hom.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        
        # Step 5: Filter by image bounds (optional)
        if filter_bounds:
            if image_shape is None:
                # Use calibration image size if not provided
                w, h = cam_intrinsics['image_size']
            else:
                h, w = image_shape[:2]
            
            u_valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w)
            v_valid = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
            in_bounds = u_valid & v_valid
            
            points_2d = points_2d[in_bounds]
            depths = depths[in_bounds]
            valid_indices = valid_indices[in_bounds]
        
        return points_2d, depths, valid_indices
    
    # =========================================================================
    # QUALITY ANALYSIS METHODS
    # =========================================================================
    
    def _compute_region_statistics(
        self,
        points_2d: np.ndarray,
        depths: np.ndarray,
        image_height: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for predefined image regions.
        
        Divides image into top/middle/bottom thirds.
        """
        region_stats = {}
        
        for region_name, (y_min_frac, y_max_frac) in self.IMAGE_REGIONS.items():
            y_min = int(y_min_frac * image_height)
            y_max = int(y_max_frac * image_height)
            
            in_region = (points_2d[:, 1] >= y_min) & (points_2d[:, 1] < y_max)
            count = int(np.sum(in_region))
            
            if count > 0:
                percentage = float((count / len(points_2d)) * 100)
                avg_depth = float(depths[in_region].mean())
                std_depth = float(depths[in_region].std())
            else:
                percentage = 0.0
                avg_depth = 0.0
                std_depth = 0.0
            
            region_stats[region_name] = {
                'point_count': count,
                'percentage': percentage,
                'avg_depth': avg_depth,
                'std_depth': std_depth
            }
        
        return region_stats
    
    def analyze_projection_quality(
        self,
        lidar_points: np.ndarray,
        points_2d: np.ndarray,
        depths: np.ndarray,
        valid_indices: np.ndarray,
        image_shape: Tuple[int, int],
        frame_idx: int = 0,
        timestamp: float = 0.0
    ) -> Dict:
        """
        Analyze quality of LiDAR-to-camera projection.
        
        Args:
            lidar_points: Original LiDAR points (N, 4)
            points_2d: Projected 2D points (M, 2)
            depths: Depths of projected points (M,)
            valid_indices: Original indices of projected points (M,)
            image_shape: (height, width) of camera image
            frame_idx: Frame identifier
            timestamp: Timestamp in seconds
            
        Returns:
            Dictionary with projection quality metrics (JSON-serializable)
        """
        h, w = image_shape[:2]
        
        # Point counts
        total_lidar = len(lidar_points)
        points_in_fov = len(valid_indices)
        points_in_image = len(points_2d)
        
        # Projection rates
        proj_rate_fov = float((points_in_fov / total_lidar) * 100)
        proj_rate_image = float((points_in_image / total_lidar) * 100)
        
        # Depth statistics
        depth_stats = {
            'min': float(depths.min()),
            'max': float(depths.max()),
            'mean': float(depths.mean()),
            'median': float(np.median(depths)),
            'std': float(depths.std()),
            'p25': float(np.percentile(depths, 25)),
            'p75': float(np.percentile(depths, 75))
        }
        
        # Image coverage
        if len(points_2d) > 0:
            u_coverage = float((points_2d[:, 0].max() - points_2d[:, 0].min()) / w * 100)
            v_coverage = float((points_2d[:, 1].max() - points_2d[:, 1].min()) / h * 100)
        else:
            u_coverage = 0.0
            v_coverage = 0.0
        
        coverage = {
            'horizontal_pct': u_coverage,
            'vertical_pct': v_coverage
        }
        
        # Depth distribution
        depth_hist, _ = np.histogram(depths, bins=self.DEPTH_BINS)
        depth_distribution = {}
        for i in range(len(depth_hist)):
            range_key = f"{self.DEPTH_BINS[i]:.0f}-{self.DEPTH_BINS[i+1]:.0f}m"
            depth_distribution[range_key] = int(depth_hist[i])
        
        # Region analysis
        region_analysis = self._compute_region_statistics(points_2d, depths, h)
        
        # Quality flags
        quality_flags = {
            'sufficient_projection_rate': bool(proj_rate_image > 60.0),
            'good_horizontal_coverage': bool(u_coverage > 90.0),
            'good_vertical_coverage': bool(v_coverage > 50.0),
            'balanced_depth_distribution': bool(depth_stats['mean'] < 30.0)
        }
        
        metrics = ProjectionQualityMetrics(
            frame_idx=frame_idx,
            timestamp=timestamp,
            total_lidar_points=total_lidar,
            points_in_camera_fov=points_in_fov,
            points_in_image_bounds=points_in_image,
            projection_rate_fov_pct=proj_rate_fov,
            projection_rate_image_pct=proj_rate_image,
            depth_stats=depth_stats,
            coverage=coverage,
            depth_distribution=depth_distribution,
            region_analysis=region_analysis,
            quality_flags=quality_flags
        )
        
        return asdict(metrics)
    
    def analyze_region_temporal(
        self,
        fusion_results: List[Dict],
        region_box: Tuple[int, int, int, int],
        timestamps: Optional[List[float]] = None
    ) -> Dict:
        """
        Analyze a specific image region over time.
        
        Tracks point count, depth statistics, and consistency for a region
        across multiple frames.
        
        Args:
            fusion_results: List of dicts with 'frame_idx', 'points_2d', 'depths'
            region_box: (x_min, y_min, x_max, y_max) in pixels
            timestamps: Optional list of timestamps
            
        Returns:
            Dictionary with temporal region analysis (JSON-serializable)
        """
        x_min, y_min, x_max, y_max = region_box
        
        if timestamps is None:
            timestamps = [0.0] * len(fusion_results)
        
        temporal_metrics = []
        
        for data, ts in zip(fusion_results, timestamps):
            frame_idx = data['frame_idx']
            points_2d = data['points_2d']
            depths = data['depths']
            
            # Filter points in region
            in_region = (
                (points_2d[:, 0] >= x_min) & (points_2d[:, 0] <= x_max) &
                (points_2d[:, 1] >= y_min) & (points_2d[:, 1] <= y_max)
            )
            
            points_in_region = int(np.sum(in_region))
            
            if points_in_region > 0:
                avg_depth = float(depths[in_region].mean())
                depth_std = float(depths[in_region].std())
                min_depth = float(depths[in_region].min())
                max_depth = float(depths[in_region].max())
            else:
                avg_depth = 0.0
                depth_std = 0.0
                min_depth = 0.0
                max_depth = 0.0
            
            temporal_metrics.append({
                'frame': frame_idx,
                'timestamp': ts,
                'points_in_region': points_in_region,
                'avg_depth': avg_depth,
                'depth_std': depth_std,
                'min_depth': min_depth,
                'max_depth': max_depth
            })
        
        # Consistency analysis
        point_counts = [m['points_in_region'] for m in temporal_metrics]
        avg_depths = [m['avg_depth'] for m in temporal_metrics if m['avg_depth'] > 0]
        
        point_count_variation = (np.std(point_counts) / np.mean(point_counts) * 100) if np.mean(point_counts) > 0 else 0
        depth_variation = (np.std(avg_depths) / np.mean(avg_depths) * 100) if len(avg_depths) > 0 and np.mean(avg_depths) > 0 else 0
        
        consistency_flags = {
            'stable_point_count': bool(point_count_variation < 20.0),
            'stable_depth': bool(depth_variation < 10.0),
            'no_occlusion_events': bool(min(point_counts) > 0)
        }
        
        return {
            'region_box': list(region_box),
            'num_frames': len(fusion_results),
            'frame_indices': [d['frame_idx'] for d in fusion_results],
            'temporal_metrics': temporal_metrics,
            'aggregate': {
                'point_count_mean': float(np.mean(point_counts)),
                'point_count_std': float(np.std(point_counts)),
                'point_count_variation_pct': float(point_count_variation),
                'depth_mean': float(np.mean(avg_depths)) if avg_depths else 0.0,
                'depth_std': float(np.std(avg_depths)) if avg_depths else 0.0,
                'depth_variation_pct': float(depth_variation)
            },
            'consistency_flags': consistency_flags
        }
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def visualize_fusion(
        self,
        image: np.ndarray,
        points_2d: np.ndarray,
        depths: np.ndarray,
        frame_idx: int = 0,
        depth_range: Tuple[float, float] = (0, 50),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize LiDAR points projected onto camera image (single frame).
        
        Args:
            image: Camera image (H, W, 3) RGB or BGR
            points_2d: Projected 2D points (N, 2) [u, v]
            depths: Depth values (N,)
            frame_idx: Frame identifier
            depth_range: (min, max) for depth color scaling
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3:
            # Assume BGR from OpenCV, convert to RGB
            try:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                img_rgb = image  # Already RGB
        else:
            img_rgb = image
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 6))
        
        # Display image
        ax.imshow(img_rgb)
        
        # Overlay LiDAR points, colored by depth
        scatter = ax.scatter(
            points_2d[:, 0], points_2d[:, 1],
            c=depths,
            s=2,
            cmap='jet',  # Blue = close, Red = far
            alpha=0.7,
            vmin=depth_range[0],
            vmax=depth_range[1]
        )
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Distance (meters)',
                           fraction=0.046, pad=0.04)
        
        # Title with statistics
        ax.set_title(
            f'LiDAR-Camera Fusion - Frame {frame_idx}\n'
            f'{len(points_2d):,} points | '
            f'Depth: {depths.mean():.1f}m avg [{depths.min():.1f}, {depths.max():.1f}]m',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Image Width (pixels)', fontsize=11)
        ax.set_ylabel('Image Height (pixels)', fontsize=11)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved fusion visualization to {save_path}")
        
        return fig
    
    def visualize_multiframe(
        self,
        fusion_results: List[Dict],
        depth_range: Tuple[float, float] = (0, 50),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create side-by-side comparison of multiple frames.
        
        Args:
            fusion_results: List of dicts with 'frame_idx', 'image', 'points_2d', 'depths'
            depth_range: (min, max) for depth color scaling
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        n_frames = len(fusion_results)
        
        fig, axes = plt.subplots(1, n_frames, figsize=(6*n_frames, 6))
        
        if n_frames == 1:
            axes = [axes]
        
        for idx, data in enumerate(fusion_results):
            ax = axes[idx]
            
            # Convert image to RGB if needed
            img = data['image']
            if len(img.shape) == 3:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    img_rgb = img
            else:
                img_rgb = img
            
            # Display image
            ax.imshow(img_rgb)
            
            # Normalize depths for colormap
            depths_norm = np.clip(data['depths'], depth_range[0], depth_range[1])
            depths_norm = (depths_norm - depth_range[0]) / (depth_range[1] - depth_range[0])
            
            # Color points by depth
            colors = cm.jet(depths_norm)
            
            # Plot points
            ax.scatter(data['points_2d'][:, 0], data['points_2d'][:, 1],
                      c=colors, s=2, alpha=0.8, edgecolors='none')
            
            # Title
            ax.set_title(f'Frame {data["frame_idx"]}\n{len(data["points_2d"]):,} points',
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Single colorbar for all subplots
        sm = cm.ScalarMappable(cmap='jet',
                              norm=plt.Normalize(vmin=depth_range[0], vmax=depth_range[1]))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label('Depth (meters)', rotation=270, labelpad=20, fontsize=12)
        
        plt.suptitle('Multi-Frame Sensor Fusion Comparison',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved multi-frame visualization to {save_path}")
        
        return fig
    
    def visualize_temporal(
        self,
        fusion_results: List[Dict],
        region_box: Optional[Tuple[int, int, int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive temporal analysis visualization.
        
        Shows:
        - Top row: Original images with optional region box
        - Middle row: LiDAR overlay with region box
        - Bottom row: Temporal metrics (speed, point count, heading)
        
        Args:
            fusion_results: List of dicts with 'frame_idx', 'image', 'points_2d', 
                           'depths', and optionally 'speed_kmh', 'yaw_deg'
            region_box: Optional (x_min, y_min, x_max, y_max) to highlight
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        n_frames = len(fusion_results)
        
        # Create figure with 3 rows
        fig = plt.figure(figsize=(4*n_frames, 12))
        
        # ===== ROW 1: Original Images =====
        for i, data in enumerate(fusion_results):
            ax = plt.subplot(3, n_frames, i+1)
            
            img = data['image']
            if len(img.shape) == 3:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    img_rgb = img
            else:
                img_rgb = img
            
            ax.imshow(img_rgb)
            
            # Draw region box if provided
            if region_box:
                x_min, y_min, x_max, y_max = region_box
                rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                               linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            
            # Title with speed if available
            title = f"Frame {data['frame_idx']}"
            if 'speed_kmh' in data:
                title += f"\n{data['speed_kmh']:.1f} km/h"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # ===== ROW 2: LiDAR Overlay =====
        for i, data in enumerate(fusion_results):
            ax = plt.subplot(3, n_frames, n_frames + i+1)
            
            img = data['image']
            if len(img.shape) == 3:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    img_rgb = img
            else:
                img_rgb = img
            
            ax.imshow(img_rgb)
            
            # Color by depth
            depths_norm = np.clip(data['depths'], 0, 50) / 50
            colors = cm.jet(depths_norm)
            ax.scatter(data['points_2d'][:, 0], data['points_2d'][:, 1],
                      c=colors, s=1, alpha=0.7)
            
            # Draw region box if provided
            if region_box:
                x_min, y_min, x_max, y_max = region_box
                rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                               linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            
            ax.set_title(f"{len(data['points_2d']):,} points", fontsize=10)
            ax.axis('off')
        
        # ===== ROW 3: Temporal Metrics =====
        frames = [d['frame_idx'] for d in fusion_results]
        point_counts = [len(d['points_2d']) for d in fusion_results]
        
        # Point count over time
        ax1 = plt.subplot(3, 3, 7)
        ax1.plot(frames, point_counts, 'g-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Frame', fontsize=10)
        ax1.set_ylabel('Valid Points', fontsize=10, color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('LiDAR Point Count', fontsize=11, fontweight='bold')
        
        # Speed (if available)
        if 'speed_kmh' in fusion_results[0]:
            speeds = [d['speed_kmh'] for d in fusion_results]
            ax2 = plt.subplot(3, 3, 8)
            ax2.plot(frames, speeds, 'b-o', linewidth=2, markersize=8)
            ax2.set_xlabel('Frame', fontsize=10)
            ax2.set_ylabel('Speed (km/h)', fontsize=10, color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Vehicle Speed', fontsize=11, fontweight='bold')
        
        # Heading (if available)
        if 'yaw_deg' in fusion_results[0]:
            yaws = [d['yaw_deg'] for d in fusion_results]
            ax3 = plt.subplot(3, 3, 9)
            ax3.plot(frames, yaws, 'r-o', linewidth=2, markersize=8)
            ax3.set_xlabel('Frame', fontsize=10)
            ax3.set_ylabel('Heading (degrees)', fontsize=10, color='r')
            ax3.tick_params(axis='y', labelcolor='r')
            ax3.grid(True, alpha=0.3)
            ax3.set_title('Vehicle Heading', fontsize=11, fontweight='bold')
        
        plt.suptitle('Temporal Sensor Fusion Analysis',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved temporal visualization to {save_path}")
        
        return fig
    
    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def export_metrics(self, output_path: str, metrics: Dict) -> None:
        """
        Export projection/fusion metrics to JSON file.
        
        Args:
            output_path: Path to output JSON file
            metrics: Metrics dictionary from analyze_* methods
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        clean_result = convert_numpy_types(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(clean_result, f, indent=2)
        
        print(f"✓ Exported fusion metrics to {output_path}")


# ============================================================================
# EXAMPLE USAGE WITH KITTI
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Import CalibrationManager (assumes it's in the same directory or installed)
    # If running as standalone, you may need to adjust import path
    try:
        from CalibrationManager import CalibrationManager
    except ImportError:
        print("WARNING: CalibrationManager not found. Using placeholder.")
        print("Make sure calibration_manager.py is in the same directory.")
        CalibrationManager = None
    
    BASE_PATH = Path("raw_data_downloader/2011_09_26")
    DRIVE = "2011_09_26_drive_0001_sync"
    
    print("="*70)
    print("SENSOR FUSION MODULE - FULL DEMO")
    print("="*70)
    
    # ===== STEP 1: Initialize Calibration =====
    print("\n### STEP 1: LOAD CALIBRATION ###")
    
    if CalibrationManager is not None:
        calib_mgr = CalibrationManager(BASE_PATH)
        calib_mgr.load_calibration()
        
        # Initialize fusion
        fusion = SensorFusion(calib_mgr)
        print("✓ SensorFusion initialized")
    else:
        print("ERROR: CalibrationManager not available. Exiting demo.")
        exit(1)
    
    # ===== STEP 2: Load Single Frame Data =====
    print("\n### STEP 2: LOAD FRAME 0 DATA ###")
    
    # Load LiDAR
    lidar_file = BASE_PATH / DRIVE / 'velodyne_points' / 'data' / '0000000000.bin'
    lidar_points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    print(f"✓ LiDAR: {len(lidar_points):,} points")
    
    # Load Camera
    img_file = BASE_PATH / DRIVE / 'image_02' / 'data' / '0000000000.png'
    image = cv2.imread(str(img_file))
    print(f"✓ Camera: {image.shape}")
    
    # ===== STEP 3: Project LiDAR to Camera =====
    print("\n### STEP 3: PROJECT LIDAR TO CAMERA ###")
    
    points_2d, depths, indices = fusion.project_lidar_to_camera(
        lidar_points, 
        camera_id='02', 
        filter_bounds=True, 
        image_shape=image.shape
    )
    
    print(f"✓ Projection complete:")
    print(f"  Total LiDAR points: {len(lidar_points):,}")
    print(f"  Points in camera FOV: {len(indices):,}")
    print(f"  Points in image bounds: {len(points_2d):,}")
    print(f"  Projection rate: {len(points_2d)/len(lidar_points)*100:.1f}%")
    
    # ===== STEP 4: Analyze Quality =====
    print("\n### STEP 4: ANALYZE PROJECTION QUALITY ###")
    
    metrics = fusion.analyze_projection_quality(
        lidar_points, points_2d, depths, indices, 
        image.shape, frame_idx=0, timestamp=0.0
    )
    
    print(f"✓ Quality analysis:")
    print(f"  Projection rate (FOV): {metrics['projection_rate_fov_pct']:.1f}%")
    print(f"  Projection rate (image): {metrics['projection_rate_image_pct']:.1f}%")
    print(f"  Horizontal coverage: {metrics['coverage']['horizontal_pct']:.1f}%")
    print(f"  Vertical coverage: {metrics['coverage']['vertical_pct']:.1f}%")
    print(f"  Quality flags: {metrics['quality_flags']}")
    
    fusion.export_metrics("outputs/fusion_quality.json", metrics)
    
    # ===== STEP 5: Visualize Single Frame =====
    print("\n### STEP 5: VISUALIZE FUSION ###")
    
    fig = fusion.visualize_fusion(
        image, points_2d, depths, 
        frame_idx=0,
        depth_range=(0, 50),
        save_path="outputs/fusion_frame_0.png"
    )
    plt.show()
    
    # ===== STEP 6: Multi-Frame Analysis =====
    print("\n### STEP 6: MULTI-FRAME COMPARISON ###")
    
    # Load multiple frames
    frame_indices = [0, 50, 100]
    fusion_results = []
    
    for frame_idx in frame_indices:
        # Load data
        lidar_file = BASE_PATH / DRIVE / 'velodyne_points' / 'data' / f'{frame_idx:010d}.bin'
        lidar = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        
        img_file = BASE_PATH / DRIVE / 'image_02' / 'data' / f'{frame_idx:010d}.png'
        img = cv2.imread(str(img_file))
        
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
        
        print(f"  Frame {frame_idx}: {len(pts_2d):,} points")
    
    # Visualize comparison
    fig_multi = fusion.visualize_multiframe(
        fusion_results,
        depth_range=(0, 50),
        save_path="outputs/fusion_multiframe.png"
    )
    plt.show()
    
    # ===== STEP 7: Temporal Region Tracking =====
    print("\n### STEP 7: TEMPORAL REGION TRACKING ###")
    
    # Define region of interest (road area)
    region_box = (400, 200, 800, 350)  # (x_min, y_min, x_max, y_max)
    
    region_metrics = fusion.analyze_region_temporal(
        fusion_results,
        region_box=region_box,
        timestamps=[0.0, 5.0, 10.0]
    )
    
    print(f"✓ Region tracking:")
    print(f"  Region box: {region_box}")
    print(f"  Frames analyzed: {region_metrics['num_frames']}")
    print(f"  Point count variation: {region_metrics['aggregate']['point_count_variation_pct']:.1f}%")
    print(f"  Consistency flags: {region_metrics['consistency_flags']}")
    
    fusion.export_metrics("outputs/region_temporal.json", region_metrics)
    
    # Visualize temporal analysis
    fig_temporal = fusion.visualize_temporal(
        fusion_results,
        region_box=region_box,
        save_path="outputs/fusion_temporal.png"
    )
    plt.show()
    
    print("\n" + "="*70)
    print("✓ SENSOR FUSION DEMO COMPLETE")
    print("="*70)
    print("\nGenerated outputs:")
    print("  - outputs/fusion_quality.json")
    print("  - outputs/fusion_frame_0.png")
    print("  - outputs/fusion_multiframe.png")
    print("  - outputs/region_temporal.json")
    print("  - outputs/fusion_temporal.png")