"""
Generic GPS/IMU sensor data quality analyzer with visualization support.

Hardware-agnostic analyzer for GPS/IMU data from any source.
Provides trajectory, velocity, orientation, and acceleration metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
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
class GPSIMUFrameMetrics:
    """Metrics for a single GPS/IMU measurement."""
    frame_idx: int
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    position_local: Optional[Dict[str, float]]
    roll: float
    pitch: float
    yaw: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    velocity_north: float
    velocity_east: float
    velocity_forward: float
    velocity_left: float
    velocity_up: float
    speed_2d: float
    speed_3d: float
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float
    acceleration_forward: float
    acceleration_left: float
    acceleration_up: float
    angular_rate_x: Optional[float]
    angular_rate_y: Optional[float]
    angular_rate_z: Optional[float]
    position_accuracy: Optional[float]
    velocity_accuracy: Optional[float]
    num_satellites: Optional[int]


@dataclass
class GPSIMUSequenceMetrics:
    """Aggregate metrics for a GPS/IMU sequence."""
    num_frames: int
    frame_indices: List[int]
    duration_sec: float
    total_distance_m: float
    max_displacement_m: float
    altitude_change_m: float
    position_range: Dict[str, Dict[str, float]]
    speed_stats: Dict[str, float]
    orientation_range: Dict[str, float]
    heading_change_deg: float
    acceleration_stats: Dict[str, Dict[str, float]]
    motion_flags: Dict[str, bool]
    quality_flags: Dict[str, bool]


class GPSIMUDataAnalyzer:
    """
    Generic GPS/IMU data quality analyzer.
    
    Hardware-agnostic analyzer that accepts GPS/IMU measurements directly.
    
    Example:
        >>> analyzer = GPSIMUDataAnalyzer()
        >>> gps_imu_data = {'lat': 49.01, 'lon': 8.12, ...}
        >>> metrics = analyzer.analyze_frame(gps_imu_data, 0, 0.0)
        >>> 
        >>> # Sequence visualization (requires multiple frames)
        >>> seq_result = analyzer.analyze_sequence(data_list, indices, timestamps)
        >>> fig = analyzer.visualize_sequence(seq_result)
        >>> plt.show()
    """
    
    DEFAULT_THRESHOLDS = {
        'max_speed_kmh': 150.0,
        'max_acceleration_ms2': 10.0,
        'min_satellites': 4,
        'position_accuracy_max_m': 5.0
    }
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """Initialize GPS/IMU analyzer."""
        self.thresholds = thresholds if thresholds else self.DEFAULT_THRESHOLDS.copy()
        self.origin_lat: Optional[float] = None
        self.origin_lon: Optional[float] = None
        self.origin_alt: Optional[float] = None
    
    def _gps_to_local_enu(self, lat: float, lon: float, alt: float) -> Dict[str, float]:
        """Convert GPS to local ENU coordinates."""
        if self.origin_lat is None:
            self.origin_lat = lat
            self.origin_lon = lon
            self.origin_alt = alt
        
        north_m = (lat - self.origin_lat) * 111320.0
        east_m = (lon - self.origin_lon) * (111320.0 * np.cos(np.radians(self.origin_lat)))
        up_m = alt - self.origin_alt
        
        return {'east_m': float(east_m), 'north_m': float(north_m), 'up_m': float(up_m)}
    
    def _compute_speed(self, vn: float, ve: float, vu: float) -> Tuple[float, float]:
        """Compute 2D and 3D speed."""
        speed_2d = float(np.sqrt(vn**2 + ve**2))
        speed_3d = float(np.sqrt(vn**2 + ve**2 + vu**2))
        return speed_2d, speed_3d
    
    def analyze_frame(self, gps_imu_data: Dict, frame_idx: int = 0, 
                     timestamp: float = 0.0) -> Dict:
        """Analyze a single GPS/IMU measurement."""
        required = ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw',
                   'vn', 've', 'vf', 'vl', 'vu',
                   'ax', 'ay', 'az', 'af', 'al', 'au']
        
        for field in required:
            if field not in gps_imu_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Extract data
        lat, lon, alt = float(gps_imu_data['lat']), float(gps_imu_data['lon']), float(gps_imu_data['alt'])
        roll, pitch, yaw = float(gps_imu_data['roll']), float(gps_imu_data['pitch']), float(gps_imu_data['yaw'])
        vn, ve, vf = float(gps_imu_data['vn']), float(gps_imu_data['ve']), float(gps_imu_data['vf'])
        vl, vu = float(gps_imu_data['vl']), float(gps_imu_data['vu'])
        ax, ay, az = float(gps_imu_data['ax']), float(gps_imu_data['ay']), float(gps_imu_data['az'])
        af, al, au = float(gps_imu_data['af']), float(gps_imu_data['al']), float(gps_imu_data['au'])
        
        wx = float(gps_imu_data.get('wx', 0.0)) if 'wx' in gps_imu_data else None
        wy = float(gps_imu_data.get('wy', 0.0)) if 'wy' in gps_imu_data else None
        wz = float(gps_imu_data.get('wz', 0.0)) if 'wz' in gps_imu_data else None
        
        pos_accuracy = float(gps_imu_data['pos_accuracy']) if 'pos_accuracy' in gps_imu_data else None
        vel_accuracy = float(gps_imu_data['vel_accuracy']) if 'vel_accuracy' in gps_imu_data else None
        num_sats = int(gps_imu_data['num_sats']) if 'num_sats' in gps_imu_data else None
        
        position_local = self._gps_to_local_enu(lat, lon, alt)
        speed_2d, speed_3d = self._compute_speed(vn, ve, vu)
        
        return {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'position_local': position_local,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'roll_deg': float(np.rad2deg(roll)),
            'pitch_deg': float(np.rad2deg(pitch)),
            'yaw_deg': float(np.rad2deg(yaw)),
            'velocity_north': vn,
            'velocity_east': ve,
            'velocity_forward': vf,
            'velocity_left': vl,
            'velocity_up': vu,
            'speed_2d': speed_2d,
            'speed_3d': speed_3d,
            'acceleration_x': ax,
            'acceleration_y': ay,
            'acceleration_z': az,
            'acceleration_forward': af,
            'acceleration_left': al,
            'acceleration_up': au,
            'angular_rate_x': wx,
            'angular_rate_y': wy,
            'angular_rate_z': wz,
            'position_accuracy': pos_accuracy,
            'velocity_accuracy': vel_accuracy,
            'num_satellites': num_sats
        }
    
    def analyze_sequence(self, gps_imu_sequence: List[Dict],
                        frame_indices: Optional[List[int]] = None,
                        timestamps: Optional[List[float]] = None) -> Dict:
        """Analyze a sequence of GPS/IMU measurements."""
        if not gps_imu_sequence:
            raise ValueError("GPS/IMU sequence cannot be empty")
        
        n = len(gps_imu_sequence)
        if frame_indices is None:
            frame_indices = list(range(n))
        if timestamps is None:
            timestamps = list(range(n))
        
        # Reset origin
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None
        
        # Analyze each frame
        frame_metrics = []
        for data, idx, ts in zip(gps_imu_sequence, frame_indices, timestamps):
            metrics = self.analyze_frame(data, idx, ts)
            frame_metrics.append(metrics)
        
        # Extract arrays
        lats = np.array([m['latitude'] for m in frame_metrics])
        lons = np.array([m['longitude'] for m in frame_metrics])
        alts = np.array([m['altitude'] for m in frame_metrics])
        
        east = np.array([m['position_local']['east_m'] for m in frame_metrics])
        north = np.array([m['position_local']['north_m'] for m in frame_metrics])
        
        speeds_2d = np.array([m['speed_2d'] for m in frame_metrics])
        
        roll_deg = np.array([m['roll_deg'] for m in frame_metrics])
        pitch_deg = np.array([m['pitch_deg'] for m in frame_metrics])
        yaw_deg = np.array([m['yaw_deg'] for m in frame_metrics])
        
        ax_arr = np.array([m['acceleration_x'] for m in frame_metrics])
        ay_arr = np.array([m['acceleration_y'] for m in frame_metrics])
        az_arr = np.array([m['acceleration_z'] for m in frame_metrics])
        
        # Trajectory statistics
        east_diff = np.diff(east)
        north_diff = np.diff(north)
        segment_distances = np.sqrt(east_diff**2 + north_diff**2)
        total_distance = float(np.sum(segment_distances))
        
        displacements = np.sqrt(east**2 + north**2)
        max_displacement = float(np.max(displacements))
        altitude_change = float(alts.max() - alts.min())
        
        # Speed statistics
        speed_stats = {
            'min_ms': float(speeds_2d.min()),
            'max_ms': float(speeds_2d.max()),
            'mean_ms': float(speeds_2d.mean()),
            'std_ms': float(speeds_2d.std()),
            'min_kmh': float(speeds_2d.min() * 3.6),
            'max_kmh': float(speeds_2d.max() * 3.6),
            'mean_kmh': float(speeds_2d.mean() * 3.6)
        }
        
        orientation_range = {
            'roll_min_deg': float(roll_deg.min()),
            'roll_max_deg': float(roll_deg.max()),
            'roll_range_deg': float(roll_deg.max() - roll_deg.min()),
            'pitch_min_deg': float(pitch_deg.min()),
            'pitch_max_deg': float(pitch_deg.max()),
            'pitch_range_deg': float(pitch_deg.max() - pitch_deg.min()),
            'yaw_min_deg': float(yaw_deg.min()),
            'yaw_max_deg': float(yaw_deg.max()),
            'yaw_range_deg': float(yaw_deg.max() - yaw_deg.min())
        }
        
        heading_change = float(yaw_deg.max() - yaw_deg.min())
        
        acceleration_stats = {
            'x': {'min': float(ax_arr.min()), 'max': float(ax_arr.max()),
                  'mean': float(ax_arr.mean()), 'std': float(ax_arr.std())},
            'y': {'min': float(ay_arr.min()), 'max': float(ay_arr.max()),
                  'mean': float(ay_arr.mean()), 'std': float(ay_arr.std())},
            'z': {'min': float(az_arr.min()), 'max': float(az_arr.max()),
                  'mean': float(az_arr.mean()), 'std': float(az_arr.std())}
        }
        
        motion_flags = {
            'has_significant_motion': bool(total_distance > 10.0),
            'has_turns': bool(heading_change > 30.0),
            'has_stops': bool(speeds_2d.min() < 0.5),
            'has_high_speed': bool(speeds_2d.max() > 10.0),
            'altitude_variation': bool(abs(altitude_change) > 2.0)
        }
        
        quality_flags = {
            'reasonable_speeds': bool(speeds_2d.max() * 3.6 < self.thresholds['max_speed_kmh']),
            'reasonable_accelerations': bool(
                abs(ax_arr).max() < self.thresholds['max_acceleration_ms2'] and
                abs(ay_arr).max() < self.thresholds['max_acceleration_ms2']
            ),
            'smooth_trajectory': bool(np.std(segment_distances) / np.mean(segment_distances) < 0.5)
        }
        
        duration_sec = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        
        summary = {
            'num_frames': n,
            'frame_indices': frame_indices,
            'duration_sec': duration_sec,
            'total_distance_m': total_distance,
            'max_displacement_m': max_displacement,
            'altitude_change_m': altitude_change,
            'position_range': {
                'latitude': {'min': float(lats.min()), 'max': float(lats.max())},
                'longitude': {'min': float(lons.min()), 'max': float(lons.max())},
                'altitude': {'min': float(alts.min()), 'max': float(alts.max())}
            },
            'speed_stats': speed_stats,
            'orientation_range': orientation_range,
            'heading_change_deg': heading_change,
            'acceleration_stats': acceleration_stats,
            'motion_flags': motion_flags,
            'quality_flags': quality_flags
        }
        
        return {
            'analysis_type': 'sequence',
            'summary': summary,
            'frames': frame_metrics
        }
    
    def visualize_sequence(self, analysis_result: Dict,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 6-panel visualization of GPS/IMU sequence.
        
        Panels:
        1. 2D Trajectory (Bird's Eye View)
        2. Speed Profile
        3. Altitude Profile
        4. Heading (Yaw)
        5. Roll & Pitch
        6. Accelerations
        
        Args:
            analysis_result: Result from analyze_sequence()
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if analysis_result['analysis_type'] != 'sequence':
            raise ValueError("visualize_sequence requires sequence analysis result")
        
        frames = analysis_result['frames']
        summary = analysis_result['summary']
        
        # Extract data arrays
        time = np.array([f['timestamp'] for f in frames])
        east = np.array([f['position_local']['east_m'] for f in frames])
        north = np.array([f['position_local']['north_m'] for f in frames])
        alt = np.array([f['altitude'] for f in frames])
        speed = np.array([f['speed_2d'] for f in frames])
        yaw_deg = np.array([f['yaw_deg'] for f in frames])
        roll_deg = np.array([f['roll_deg'] for f in frames])
        pitch_deg = np.array([f['pitch_deg'] for f in frames])
        ax = np.array([f['acceleration_x'] for f in frames])
        ay = np.array([f['acceleration_y'] for f in frames])
        az = np.array([f['acceleration_z'] for f in frames])
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ========== PANEL 1: 2D Trajectory ==========
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(east, north, c=range(len(east)), cmap='viridis', s=50, alpha=0.7)
        ax1.plot(east, north, 'k-', alpha=0.3, linewidth=1)
        ax1.scatter(east[0], north[0], c='green', s=200, marker='o',
                   edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax1.scatter(east[-1], north[-1], c='red', s=200, marker='s',
                   edgecolors='black', linewidths=2, label='End', zorder=5)
        ax1.set_xlabel('East (m)', fontsize=12)
        ax1.set_ylabel('North (m)', fontsize=12)
        ax1.set_title('2D Trajectory (GPS)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Frame Number')
        
        # ========== PANEL 2: Speed Profile ==========
        ax2 = axes[0, 1]
        ax2.plot(time, speed, linewidth=2, color='blue')
        ax2.fill_between(time, speed, alpha=0.3, color='lightblue')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Speed (m/s)', fontsize=12)
        ax2.set_title('Vehicle Speed Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(speed.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {speed.mean():.2f} m/s')
        ax2.legend()
        
        # Secondary y-axis for km/h
        ax2_kmh = ax2.twinx()
        ax2_kmh.set_ylabel('Speed (km/h)', fontsize=12)
        ax2_kmh.set_ylim(ax2.get_ylim()[0]*3.6, ax2.get_ylim()[1]*3.6)
        
        # ========== PANEL 3: Altitude Profile ==========
        ax3 = axes[0, 2]
        ax3.plot(time, alt, linewidth=2, color='brown')
        ax3.fill_between(time, alt, alpha=0.3, color='tan')
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Altitude (m)', fontsize=12)
        ax3.set_title('Altitude Profile', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        altitude_change = alt.max() - alt.min()
        ax3.text(0.02, 0.98, f'Total change: {altitude_change:.2f} m',
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ========== PANEL 4: Heading (Yaw) ==========
        ax4 = axes[1, 0]
        ax4.plot(time, yaw_deg, linewidth=2, color='green')
        ax4.set_xlabel('Time (seconds)', fontsize=12)
        ax4.set_ylabel('Heading (degrees)', fontsize=12)
        ax4.set_title('Vehicle Heading (Yaw Angle)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        heading_change = yaw_deg.max() - yaw_deg.min()
        ax4.text(0.02, 0.98, f'Total rotation: {heading_change:.1f}°',
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # ========== PANEL 5: Roll & Pitch ==========
        ax5 = axes[1, 1]
        ax5.plot(time, roll_deg, linewidth=2, label='Roll', color='blue')
        ax5.plot(time, pitch_deg, linewidth=2, label='Pitch', color='red')
        ax5.set_xlabel('Time (seconds)', fontsize=12)
        ax5.set_ylabel('Angle (degrees)', fontsize=12)
        ax5.set_title('Vehicle Orientation (Roll & Pitch)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # ========== PANEL 6: Accelerations ==========
        ax6 = axes[1, 2]
        ax6.plot(time, ax, linewidth=2, label='Longitudinal (ax)', color='red', alpha=0.7)
        ax6.plot(time, ay, linewidth=2, label='Lateral (ay)', color='blue', alpha=0.7)
        ax6.plot(time, az, linewidth=2, label='Vertical (az)', color='green', alpha=0.7)
        ax6.set_xlabel('Time (seconds)', fontsize=12)
        ax6.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax6.set_title('Vehicle Accelerations', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        ax6.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Overall title
        quality_str = "✓ GOOD" if all(summary['quality_flags'].values()) else "⚠ CHECK"
        fig.suptitle(f'GPS/IMU Sequence Visualization {quality_str}\n'
                    f'Distance: {summary["total_distance_m"]:.1f}m | '
                    f'Duration: {summary["duration_sec"]:.1f}s | '
                    f'Avg Speed: {summary["speed_stats"]["mean_kmh"]:.1f} km/h',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        return fig
    
    def export_analysis(self, output_path: str, analysis_result: Dict) -> None:
        """Export analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        clean_result = convert_numpy_types(analysis_result)
        with open(output_path, 'w') as f:
            json.dump(clean_result, f, indent=2)
        
        print(f"✓ Exported GPS/IMU analysis to {output_path}")


# ============================================================================
# KITTI LOADER (Usage Example)
# ============================================================================

class KITTIOXTSLoader:
    """KITTI-specific GPS/IMU (OXTS) data loader."""
    
    def __init__(self, drive_path: str):
        self.drive_path = Path(drive_path)
        self.oxts_path = self.drive_path / 'oxts' / 'data'
        
        if not self.oxts_path.exists():
            raise FileNotFoundError(f"OXTS data not found: {self.oxts_path}")
    
    def _parse_oxts_line(self, line: str) -> Dict:
        """Parse OXTS line to dictionary."""
        values = [float(x) for x in line.strip().split()]
        return {
            'lat': values[0], 'lon': values[1], 'alt': values[2],
            'roll': values[3], 'pitch': values[4], 'yaw': values[5],
            'vn': values[6], 've': values[7], 'vf': values[8],
            'vl': values[9], 'vu': values[10],
            'ax': values[11], 'ay': values[12], 'az': values[13],
            'af': values[14], 'al': values[15], 'au': values[16],
            'wx': values[17], 'wy': values[18], 'wz': values[19],
            'pos_accuracy': values[23], 'vel_accuracy': values[24],
            'num_sats': int(values[26])
        }
    
    def load_frame(self, frame_idx: int) -> Dict:
        """Load a single OXTS frame."""
        oxts_file = self.oxts_path / f'{frame_idx:010d}.txt'
        if not oxts_file.exists():
            raise FileNotFoundError(f"Frame {frame_idx} not found")
        
        with open(oxts_file, 'r') as f:
            return self._parse_oxts_line(f.read())
    
    def load_sequence(self, start_idx: int = 0, 
                     end_idx: Optional[int] = None) -> Tuple[List[Dict], List[int]]:
        """Load a sequence of OXTS frames."""
        oxts_files = sorted(self.oxts_path.glob('*.txt'))
        available_indices = [int(f.stem) for f in oxts_files]
        
        if end_idx is None:
            end_idx = max(available_indices) + 1
        
        frame_indices = [idx for idx in available_indices if start_idx <= idx < end_idx]
        data_sequence = [self.load_frame(idx) for idx in frame_indices]
        
        return data_sequence, frame_indices


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize analyzer and loader
    analyzer = GPSIMUDataAnalyzer()
    loader = KITTIOXTSLoader("raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync")
    
    print("="*70)
    print("GPS/IMU DATA QUALITY ANALYSIS WITH VISUALIZATION")
    print("="*70)
    
    # Sequence analysis with visualization
    print("\n### ANALYZING SEQUENCE (108 frames) ###")
    data_seq, indices = loader.load_sequence(0, 108)
    timestamps = [i * 0.1 for i in range(len(indices))]  # 10 Hz
    
    # Analyze
    seq_result = analyzer.analyze_sequence(data_seq, indices, timestamps)
    summary = seq_result['summary']
    
    print(f"Frames: {summary['num_frames']}")
    print(f"Duration: {summary['duration_sec']:.1f} seconds")
    print(f"Distance: {summary['total_distance_m']:.2f} m")
    print(f"Avg speed: {summary['speed_stats']['mean_kmh']:.1f} km/h")
    print(f"Heading change: {summary['heading_change_deg']:.1f}°")
    print(f"Quality flags: {summary['quality_flags']}")
    
    # Create visualization
    fig = analyzer.visualize_sequence(seq_result,
                                     save_path="outputs/gps_imu_sequence_viz.png")
    
    # Export metrics
    analyzer.export_analysis("outputs/gps_imu_sequence.json", seq_result)
    
    print("\n✓ Analysis complete!")
    print("  - Metrics saved to: outputs/gps_imu_sequence.json")
    print("  - Visualization saved to: outputs/gps_imu_sequence_viz.png")