"""
LiDAR sensor data quality analyzer with visualization support.

Provides comprehensive point cloud analysis including:
- Spatial coverage (range, azimuth, elevation)
- Point density distribution
- Reflectance characteristics
- Resolution analysis
- Basic visualizations for quality validation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.spatial import cKDTree
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
class LiDARFrameMetrics:
    """Metrics for a single LiDAR frame."""
    frame_idx: int
    timestamp: float
    point_count: int
    range_stats: Dict[str, float]
    x_range: Dict[str, float]
    y_range: Dict[str, float]
    z_range: Dict[str, float]
    horizontal_fov_deg: float
    vertical_fov_deg: float
    azimuth_coverage: Dict[str, int]
    range_distribution: Dict[str, int]
    mean_point_density: float
    reflectance_stats: Dict[str, float]
    reflectance_zero_pct: float
    spatial_resolution_cm: float
    estimated_sensor_height_m: float
    quality_flags: Dict[str, bool]


@dataclass
class LiDARSequenceMetrics:
    """Aggregate metrics for multiple frames."""
    num_frames: int
    frame_indices: List[int]
    point_count_mean: float
    point_count_std: float
    point_count_min: int
    point_count_max: int
    point_count_variation_pct: float
    range_mean: float
    range_std: float
    coverage_consistent: bool
    resolution_consistent: bool
    quality_flags: Dict[str, bool]


class LiDARDataAnalyzer:
    """
    Analyze LiDAR point cloud data quality.
    
    Hardware-agnostic analyzer suitable for any LiDAR sensor.
    
    Example:
        >>> analyzer = LiDARDataAnalyzer()
        >>> points = analyzer.load_frame(0)  # From KITTI loader
        >>> metrics = analyzer.analyze_frame(points, frame_idx=0)
        >>> fig = analyzer.visualize_frame(points, frame_idx=0)
        >>> plt.show()
    """
    
    # Analysis configuration
    RANGE_BINS = [0, 10, 20, 30, 40, 50, 100]  # meters
    AZIMUTH_SECTORS = {
        'front': (-22.5, 22.5),
        'front_left': (22.5, 67.5),
        'left': (67.5, 112.5),
        'rear_left': (112.5, 157.5),
        'rear': (157.5, 180),
        'rear_right': (-157.5, -112.5),
        'right': (-112.5, -67.5),
        'front_right': (-67.5, -22.5)
    }
    
    def __init__(self):
        """Initialize LiDAR data analyzer."""
        pass
    
    def _compute_range_statistics(self, points: np.ndarray) -> Dict[str, float]:
        """Calculate 3D distance statistics."""
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        return {
            'min': float(distances.min()),
            'max': float(distances.max()),
            'mean': float(distances.mean()),
            'median': float(np.median(distances)),
            'std': float(distances.std()),
            'p25': float(np.percentile(distances, 25)),
            'p75': float(np.percentile(distances, 75))
        }
    
    def _compute_coordinate_statistics(self, coords: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for X, Y, or Z coordinates."""
        return {
            'min': float(coords.min()),
            'max': float(coords.max()),
            'mean': float(coords.mean()),
            'std': float(coords.std()),
            'range': float(coords.max() - coords.min())
        }
    
    def _compute_azimuth_coverage(self, points: np.ndarray) -> Dict[str, int]:
        """Calculate point distribution across azimuth sectors."""
        azimuth = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))
        
        sector_counts = {}
        for sector_name, (angle_min, angle_max) in self.AZIMUTH_SECTORS.items():
            if sector_name == 'rear':
                in_sector = (azimuth >= angle_min) | (azimuth <= -157.5)
            else:
                in_sector = (azimuth >= angle_min) & (azimuth < angle_max)
            sector_counts[sector_name] = int(np.sum(in_sector))
        
        return sector_counts
    
    def _compute_range_distribution(self, points: np.ndarray) -> Dict[str, int]:
        """Calculate point count per range bin."""
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        hist, _ = np.histogram(distances, bins=self.RANGE_BINS)
        
        distribution = {}
        for i in range(len(hist)):
            range_key = f"{self.RANGE_BINS[i]:.0f}-{self.RANGE_BINS[i+1]:.0f}m"
            distribution[range_key] = int(hist[i])
        
        return distribution
    
    def _compute_reflectance_statistics(self, points: np.ndarray) -> Tuple[Dict, float]:
        """Calculate reflectance statistics."""
        reflectance = points[:, 3]
        stats = {
            'min': float(reflectance.min()),
            'max': float(reflectance.max()),
            'mean': float(reflectance.mean()),
            'median': float(np.median(reflectance)),
            'std': float(reflectance.std())
        }
        zero_pct = float((np.sum(reflectance == 0) / len(reflectance)) * 100)
        return stats, zero_pct
    
    def _compute_spatial_resolution(self, points: np.ndarray, sample_size: int = 5000) -> float:
        """Estimate spatial resolution via nearest-neighbor distance."""
        sample_size = min(sample_size, len(points))
        sample_idx = np.random.choice(len(points), sample_size, replace=False)
        points_sample = points[sample_idx, :3]
        
        tree = cKDTree(points_sample)
        distances, _ = tree.query(points_sample, k=2)
        nearest_neighbor_dists = distances[:, 1]
        
        return float(nearest_neighbor_dists.mean() * 100)
    
    def _estimate_sensor_height(self, points: np.ndarray) -> float:
        """Estimate sensor mounting height above ground."""
        z_min = points[:, 2].min()
        return float(abs(z_min))
    
    def _compute_fov(self, points: np.ndarray) -> Tuple[float, float]:
        """Calculate horizontal and vertical field of view."""
        azimuth = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))
        horizontal_fov = azimuth.max() - azimuth.min()
        
        horizontal_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        elevation = np.rad2deg(np.arctan2(points[:, 2], horizontal_dist))
        vertical_fov = elevation.max() - elevation.min()
        
        return float(horizontal_fov), float(vertical_fov)
    
    def analyze_frame(self, points: np.ndarray, frame_idx: int = 0, 
                     timestamp: float = 0.0) -> Dict:
        """
        Comprehensive analysis of a single LiDAR frame.
        
        Args:
            points: (N, 4) array with [x, y, z, reflectance]
            frame_idx: Frame identifier
            timestamp: Frame timestamp (seconds)
            
        Returns:
            Dictionary with frame metrics (JSON-serializable)
        """
        if not isinstance(points, np.ndarray) or points.shape[1] != 4:
            raise ValueError("Points must be (N, 4) array with [x, y, z, reflectance]")
        
        # Compute all metrics
        range_stats = self._compute_range_statistics(points)
        x_stats = self._compute_coordinate_statistics(points[:, 0])
        y_stats = self._compute_coordinate_statistics(points[:, 1])
        z_stats = self._compute_coordinate_statistics(points[:, 2])
        azimuth_coverage = self._compute_azimuth_coverage(points)
        range_distribution = self._compute_range_distribution(points)
        reflectance_stats, reflectance_zero_pct = self._compute_reflectance_statistics(points)
        spatial_resolution_cm = self._compute_spatial_resolution(points)
        sensor_height = self._estimate_sensor_height(points)
        horizontal_fov, vertical_fov = self._compute_fov(points)
        
        total_points = len(points)
        mean_range = range_stats['mean']
        mean_density = total_points / (mean_range * len(self.RANGE_BINS)) if mean_range > 0 else 0
        
        # Quality flags
        quality_flags = {
            'sufficient_points': bool(total_points > 50000),
            'full_360_coverage': bool(horizontal_fov > 355),
            'good_vertical_coverage': bool(vertical_fov > 20),
            'reasonable_range': bool(range_stats['mean'] < 30),
            'sufficient_reflectance': bool(reflectance_zero_pct < 50)
        }
        
        metrics = LiDARFrameMetrics(
            frame_idx=frame_idx,
            timestamp=timestamp,
            point_count=total_points,
            range_stats=range_stats,
            x_range=x_stats,
            y_range=y_stats,
            z_range=z_stats,
            horizontal_fov_deg=horizontal_fov,
            vertical_fov_deg=vertical_fov,
            azimuth_coverage=azimuth_coverage,
            range_distribution=range_distribution,
            mean_point_density=float(mean_density),
            reflectance_stats=reflectance_stats,
            reflectance_zero_pct=reflectance_zero_pct,
            spatial_resolution_cm=spatial_resolution_cm,
            estimated_sensor_height_m=sensor_height,
            quality_flags=quality_flags
        )
        
        return asdict(metrics)
    
    def visualize_frame(self, points: np.ndarray, frame_idx: int = 0,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 4-panel visualization of LiDAR frame.
        
        Panels:
        1. 3D Point Cloud (colored by height)
        2. Bird's Eye View (BEV)
        3. Range Distribution (histogram)
        4. Azimuth Coverage (360° bar chart)
        
        Args:
            points: (N, 4) array with [x, y, z, reflectance]
            frame_idx: Frame identifier for title
            save_path: Optional path to save figure (PNG/PDF)
            
        Returns:
            Matplotlib Figure object
        """
        # Subsample for visualization (faster rendering)
        sample_size = min(15000, len(points))
        sample_idx = np.random.choice(len(points), sample_size, replace=False)
        points_sample = points[sample_idx]
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # ========== PANEL 1: 3D Point Cloud ==========
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        scatter = ax1.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                             c=points_sample[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('X (forward, m)')
        ax1.set_ylabel('Y (left, m)')
        ax1.set_zlabel('Z (up, m)')
        ax1.set_title('3D Point Cloud (colored by height)', fontweight='bold')
        ax1.set_xlim([-40, 40])
        ax1.set_ylim([-40, 40])
        ax1.set_zlim([-3, 3])
        plt.colorbar(scatter, ax=ax1, label='Height (m)', shrink=0.5, pad=0.1)
        
        # ========== PANEL 2: Bird's Eye View ==========
        ax2 = fig.add_subplot(2, 2, 2)
        scatter2 = ax2.scatter(points_sample[:, 0], points_sample[:, 1],
                              c=points_sample[:, 2], cmap='viridis', s=2, alpha=0.6)
        ax2.set_xlabel('X (forward, m)')
        ax2.set_ylabel('Y (left, m)')
        ax2.set_title("Bird's Eye View (Top-Down)", fontweight='bold')
        ax2.set_xlim([-40, 40])
        ax2.set_ylim([-40, 40])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2, label='Height (m)', shrink=0.8)
        
        # ========== PANEL 3: Range Distribution ==========
        ax3 = fig.add_subplot(2, 2, 3)
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        hist, bin_edges = np.histogram(distances, bins=self.RANGE_BINS)
        
        range_labels = [f"{self.RANGE_BINS[i]:.0f}-{self.RANGE_BINS[i+1]:.0f}m" 
                       for i in range(len(hist))]
        bars = ax3.bar(range_labels, hist, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Distance Range (m)')
        ax3.set_ylabel('Number of Points')
        ax3.set_title('Range Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, hist):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # ========== PANEL 4: Azimuth Coverage ==========
        ax4 = fig.add_subplot(2, 2, 4)
        azimuth_coverage = self._compute_azimuth_coverage(points)
        
        sector_names = ['Front', 'F-Left', 'Left', 'R-Left', 
                       'Rear', 'R-Right', 'Right', 'F-Right']
        sector_counts = [azimuth_coverage[key] for key in 
                        ['front', 'front_left', 'left', 'rear_left',
                         'rear', 'rear_right', 'right', 'front_right']]
        
        bars = ax4.bar(sector_names, sector_counts, color='teal', edgecolor='black')
        ax4.set_xlabel('Sector')
        ax4.set_ylabel('Number of Points')
        ax4.set_title('360° Azimuth Coverage', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add percentage labels
        total_points = sum(sector_counts)
        for bar, count in zip(bars, sector_counts):
            height = bar.get_height()
            pct = (count / total_points) * 100
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Overall title
        fig.suptitle(f'LiDAR Frame {frame_idx} - Quality Visualization '
                    f'({len(points):,} points)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        return fig
    
    def analyze_sequence(self, points_list: List[np.ndarray],
                        frame_indices: Optional[List[int]] = None,
                        timestamps: Optional[List[float]] = None) -> Dict:
        """
        Analyze a sequence of LiDAR frames.
        
        Args:
            points_list: List of point cloud arrays
            frame_indices: Optional list of frame identifiers
            timestamps: Optional list of timestamps
            
        Returns:
            Dictionary with sequence analysis
        """
        if not points_list:
            raise ValueError("Points list cannot be empty")
        
        n = len(points_list)
        if frame_indices is None:
            frame_indices = list(range(n))
        if timestamps is None:
            timestamps = [0.0] * n
        
        # Analyze each frame
        frame_metrics = []
        for points, idx, ts in zip(points_list, frame_indices, timestamps):
            metrics = self.analyze_frame(points, idx, ts)
            frame_metrics.append(metrics)
        
        # Aggregate statistics
        point_counts = [m['point_count'] for m in frame_metrics]
        ranges = [m['range_stats']['mean'] for m in frame_metrics]
        resolutions = [m['spatial_resolution_cm'] for m in frame_metrics]
        
        point_count_variation = (np.std(point_counts) / np.mean(point_counts)) * 100
        resolution_variation = (np.std(resolutions) / np.mean(resolutions)) * 100
        
        quality_flags = {
            'sufficient_points': bool(np.mean(point_counts) > 50000),
            'consistent_point_count': bool(point_count_variation < 5.0),
            'consistent_resolution': bool(resolution_variation < 20.0),
            'full_360_coverage': all(m['horizontal_fov_deg'] > 355 for m in frame_metrics)
        }
        
        metrics = LiDARSequenceMetrics(
            num_frames=len(frame_indices),
            frame_indices=frame_indices,
            point_count_mean=float(np.mean(point_counts)),
            point_count_std=float(np.std(point_counts)),
            point_count_min=int(np.min(point_counts)),
            point_count_max=int(np.max(point_counts)),
            point_count_variation_pct=float(point_count_variation),
            range_mean=float(np.mean(ranges)),
            range_std=float(np.std(ranges)),
            coverage_consistent=quality_flags['full_360_coverage'],
            resolution_consistent=quality_flags['consistent_resolution'],
            quality_flags=quality_flags
        )
        
        return {
            'analysis_type': 'sequence',
            'summary': asdict(metrics),
            'frames': frame_metrics
        }
    
    def export_analysis(self, output_path: str, analysis_result: Dict) -> None:
        """Export analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        clean_result = convert_numpy_types(analysis_result)
        with open(output_path, 'w') as f:
            json.dump(clean_result, f, indent=2)
        
        print(f"✓ Exported LiDAR analysis to {output_path}")


# ============================================================================
# KITTI LOADER (Usage Example)
# ============================================================================

class KITTILiDARLoader:
    """KITTI-specific LiDAR data loader."""
    
    def __init__(self, drive_path: str):
        self.drive_path = Path(drive_path)
        self.lidar_path = self.drive_path / 'velodyne_points' / 'data'
        
        if not self.lidar_path.exists():
            raise FileNotFoundError(f"LiDAR data not found: {self.lidar_path}")
    
    def load_frame(self, frame_idx: int) -> np.ndarray:
        """Load a single LiDAR frame."""
        velo_file = self.lidar_path / f'{frame_idx:010d}.bin'
        if not velo_file.exists():
            raise FileNotFoundError(f"Frame {frame_idx} not found")
        
        points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        return points
    
    def load_sequence(self, start_idx: int = 0, 
                     end_idx: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
        """Load a sequence of LiDAR frames."""
        frame_files = sorted(self.lidar_path.glob('*.bin'))
        available_indices = [int(f.stem) for f in frame_files]
        
        if end_idx is None:
            end_idx = max(available_indices) + 1
        
        frame_indices = [idx for idx in available_indices if start_idx <= idx < end_idx]
        points_list = [self.load_frame(idx) for idx in frame_indices]
        
        return points_list, frame_indices


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize analyzer and loader
    analyzer = LiDARDataAnalyzer()
    loader = KITTILiDARLoader("raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync")
    
    print("="*70)
    print("LIDAR DATA QUALITY ANALYSIS WITH VISUALIZATION")
    print("="*70)
    
    # Single frame analysis with visualization
    print("\n### ANALYZING FRAME 0 ###")
    points = loader.load_frame(0)
    print(f"Loaded {len(points):,} points")
    
    # Compute metrics
    metrics = analyzer.analyze_frame(points, frame_idx=0, timestamp=0.0)
    print(f"Point count: {metrics['point_count']:,}")
    print(f"Range: {metrics['range_stats']['mean']:.2f} m (mean)")
    print(f"Horizontal FOV: {metrics['horizontal_fov_deg']:.1f}°")
    print(f"Spatial resolution: {metrics['spatial_resolution_cm']:.2f} cm")
    
    # Create visualization
    fig = analyzer.visualize_frame(points, frame_idx=0, 
                                   save_path="outputs/lidar_frame_0_viz.png")
    
    # Export metrics
    analyzer.export_analysis("outputs/lidar_frame_0.json", metrics)
    
    print("\n✓ Analysis complete!")
    print("  - Metrics saved to: outputs/lidar_frame_0.json")
    print("  - Visualization saved to: outputs/lidar_frame_0_viz.png")