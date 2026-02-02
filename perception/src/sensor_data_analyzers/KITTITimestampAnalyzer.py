"""
Timestamp synchronization analysis for KITTI dataset.

This module provides comprehensive temporal analysis of multi-sensor data streams,
including frame rate calculation, inter-sensor offset analysis, and drift detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SensorTimingStats:
    """Statistics for a single sensor's timing characteristics."""
    sensor_name: str
    frame_count: int
    frame_rate_hz: float
    mean_interval_ms: float
    std_interval_ms: float
    min_interval_ms: float
    max_interval_ms: float


@dataclass
class SyncQualityMetrics:
    """Cross-sensor synchronization quality metrics."""
    reference_sensor: str
    offsets_mean_ms: Dict[str, float]
    offsets_std_ms: Dict[str, float]
    drift_ms: Dict[str, float]  # first to last frame
    max_jitter_ms: float


class KITTITimestampAnalyzer:
    """
    Analyze temporal synchronization of KITTI sensor streams.
    
    Provides comprehensive timing analysis including:
    - Frame rate calculation per sensor
    - Inter-frame interval statistics
    - Cross-sensor offset measurement
    - Temporal drift detection
    
    Example:
        >>> analyzer = KITTITimestampAnalyzer(
        ...     'data/2011_09_26/2011_09_26_drive_0001_sync'
        ... )
        >>> analyzer.load_all_timestamps()
        >>> stats = analyzer.analyze_frame_rates()
        >>> sync_metrics = analyzer.analyze_synchronization(reference='lidar')
        >>> analyzer.print_summary()
    """
    
    # Sensor configuration
    SENSORS = {
        'image_00': 'gray_left',
        'image_01': 'gray_right',
        'image_02': 'color_left',
        'image_03': 'color_right',
        'velodyne_points': 'lidar',
        'oxts': 'gps_imu'
    }
    
    def __init__(self, drive_path: str):
        """
        Initialize analyzer with path to KITTI drive directory.
        
        Args:
            drive_path: Path to drive folder (e.g., '2011_09_26_drive_0001_sync')
        """
        self.drive_path = Path(drive_path)
        self.timestamp_data: Dict[str, List[str]] = {}
        self.df: Optional[pd.DataFrame] = None
        self.timing_stats: Dict[str, SensorTimingStats] = {}
        self.sync_metrics: Optional[SyncQualityMetrics] = None
        
        if not self.drive_path.exists():
            raise FileNotFoundError(f"Drive path not found: {self.drive_path}")
    
    def load_timestamps(self, sensor_folder: str) -> Optional[List[str]]:
        """
        Load timestamps from a single sensor folder.
        
        Args:
            sensor_folder: Folder name (e.g., 'image_02', 'velodyne_points')
            
        Returns:
            List of timestamp strings, or None if file not found
        """
        timestamp_file = self.drive_path / sensor_folder / "timestamps.txt"
        
        if not timestamp_file.exists():
            print(f"Warning: {timestamp_file} not found!")
            return None
        
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f:
                timestamps.append(line.strip())
        
        return timestamps
    
    def load_all_timestamps(self) -> None:
        """Load timestamps from all configured sensors."""
        print("=" * 70)
        print("LOADING TIMESTAMPS")
        print("=" * 70)
        
        for folder, name in self.SENSORS.items():
            ts = self.load_timestamps(folder)
            if ts is not None:
                self.timestamp_data[name] = ts
                print(f"✓ {name:<15}: {len(ts):3d} timestamps from {folder}")
        
        print(f"\nTotal sensors loaded: {len(self.timestamp_data)}")
        
        # Build DataFrame
        self._build_dataframe()
    
    def _build_dataframe(self) -> None:
        """Construct pandas DataFrame with all timestamp data."""
        if not self.timestamp_data:
            raise ValueError("No timestamp data loaded. Call load_all_timestamps() first.")
        
        # Find maximum frame count
        max_frames = max(len(ts) for ts in self.timestamp_data.values())
        
        # Create base DataFrame
        self.df = pd.DataFrame({'frame_idx': range(max_frames)})
        
        # Add raw timestamps
        for sensor_name, timestamps in self.timestamp_data.items():
            padded_timestamps = timestamps + [None] * (max_frames - len(timestamps))
            self.df[sensor_name] = padded_timestamps
        
        # Convert to datetime
        for sensor_name in self.timestamp_data.keys():
            self.df[f'{sensor_name}_dt'] = pd.to_datetime(self.df[sensor_name])
        
        # Calculate elapsed time from start
        for sensor_name in self.timestamp_data.keys():
            self.df[f'{sensor_name}_elapsed'] = (
                self.df[f'{sensor_name}_dt'] - self.df[f'{sensor_name}_dt'].iloc[0]
            ).dt.total_seconds()
        
        # Calculate inter-frame intervals
        for sensor_name in self.timestamp_data.keys():
            self.df[f'{sensor_name}_interval'] = self.df[f'{sensor_name}_elapsed'].diff()
    
    def check_completeness(self) -> Dict[str, int]:
        """
        Check for missing frames in each sensor stream.
        
        Returns:
            Dictionary mapping sensor name to number of missing frames
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_all_timestamps() first.")
        
        missing = {}
        for sensor_name in self.timestamp_data.keys():
            missing[sensor_name] = self.df[sensor_name].isnull().sum()
        
        return missing
    
    def analyze_frame_rates(self) -> Dict[str, SensorTimingStats]:
        """
        Calculate frame rate statistics for each sensor.
        
        Returns:
            Dictionary mapping sensor name to SensorTimingStats
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_all_timestamps() first.")
        
        for sensor_name in self.timestamp_data.keys():
            # Get interval statistics (skip first NaN value)
            intervals = self.df[f'{sensor_name}_interval'].dropna()
            
            mean_interval_sec = intervals.mean()
            frame_rate = 1.0 / mean_interval_sec if mean_interval_sec > 0 else 0
            
            self.timing_stats[sensor_name] = SensorTimingStats(
                sensor_name=sensor_name,
                frame_count=len(self.timestamp_data[sensor_name]),
                frame_rate_hz=frame_rate,
                mean_interval_ms=mean_interval_sec * 1000,
                std_interval_ms=intervals.std() * 1000,
                min_interval_ms=intervals.min() * 1000,
                max_interval_ms=intervals.max() * 1000
            )
        
        return self.timing_stats
    
    def analyze_synchronization(
        self, 
        reference: str = 'lidar'
    ) -> SyncQualityMetrics:
        """
        Analyze cross-sensor synchronization quality.
        
        Args:
            reference: Reference sensor for offset calculation (default: 'lidar')
            
        Returns:
            SyncQualityMetrics with offset and drift information
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_all_timestamps() first.")
        
        if reference not in self.timestamp_data:
            raise ValueError(f"Reference sensor '{reference}' not found in loaded data.")
        
        offsets_mean = {}
        offsets_std = {}
        drift = {}
        max_jitter = 0
        
        # Calculate offsets relative to reference sensor
        for sensor_name in self.timestamp_data.keys():
            if sensor_name != reference:
                # Calculate offset in milliseconds
                offset_col = f'{sensor_name}_vs_{reference}'
                self.df[offset_col] = (
                    self.df[f'{sensor_name}_dt'] - self.df[f'{reference}_dt']
                ).dt.total_seconds() * 1000
                
                offsets = self.df[offset_col].dropna()
                
                offsets_mean[sensor_name] = offsets.mean()
                offsets_std[sensor_name] = offsets.std()
                
                # Calculate drift (first to last frame)
                first_offset = offsets.iloc[0]
                last_offset = offsets.iloc[-1]
                drift[sensor_name] = last_offset - first_offset
                
                # Track maximum jitter
                max_jitter = max(max_jitter, offsets_std[sensor_name])
        
        self.sync_metrics = SyncQualityMetrics(
            reference_sensor=reference,
            offsets_mean_ms=offsets_mean,
            offsets_std_ms=offsets_std,
            drift_ms=drift,
            max_jitter_ms=max_jitter
        )
        
        return self.sync_metrics
    
    def get_summary_dict(self) -> Dict:
        """
        Get all analysis results as a dictionary (for JSON export).
        
        Returns:
            Dictionary with all timing and sync metrics
        """
        if not self.timing_stats:
            self.analyze_frame_rates()
        
        if not self.sync_metrics:
            self.analyze_synchronization()
        
        summary = {
            'frame_count': len(self.df),
            'duration_sec': self.df[f'{list(self.timestamp_data.keys())[0]}_elapsed'].iloc[-1],
            'sensors': {}
        }
        
        for sensor_name, stats in self.timing_stats.items():
            summary['sensors'][sensor_name] = {
                'frame_count': stats.frame_count,
                'frame_rate_hz': round(stats.frame_rate_hz, 2),
                'mean_interval_ms': round(stats.mean_interval_ms, 2),
                'std_interval_ms': round(stats.std_interval_ms, 2)
            }
        
        summary['synchronization'] = {
            'reference_sensor': self.sync_metrics.reference_sensor,
            'max_jitter_ms': round(self.sync_metrics.max_jitter_ms, 2),
            'offsets': {
                sensor: {
                    'mean_ms': round(self.sync_metrics.offsets_mean_ms[sensor], 2),
                    'std_ms': round(self.sync_metrics.offsets_std_ms[sensor], 2),
                    'drift_ms': round(self.sync_metrics.drift_ms[sensor], 2)
                }
                for sensor in self.sync_metrics.offsets_mean_ms.keys()
            }
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print comprehensive summary of timing analysis."""
        if not self.timing_stats:
            self.analyze_frame_rates()
        
        if not self.sync_metrics:
            self.analyze_synchronization()
        
        print("\n" + "=" * 70)
        print("TIMESTAMP ANALYSIS SUMMARY")
        print("=" * 70)
        
        # Frame counts
        print("\n### FRAME COUNTS ###")
        print(f"{'Sensor':<15} {'Frames':<10} {'Missing':<10}")
        print("-" * 35)
        missing = self.check_completeness()
        for sensor_name in self.timestamp_data.keys():
            frames = self.timing_stats[sensor_name].frame_count
            miss = missing[sensor_name]
            print(f"{sensor_name:<15} {frames:<10} {miss:<10}")
        
        # Frame rates
        print("\n### FRAME RATES ###")
        print(f"{'Sensor':<15} {'Rate (Hz)':<12} {'Interval (ms)':<15} {'Std (ms)':<10}")
        print("-" * 52)
        for sensor_name, stats in self.timing_stats.items():
            print(f"{sensor_name:<15} {stats.frame_rate_hz:<12.2f} "
                  f"{stats.mean_interval_ms:<15.2f} {stats.std_interval_ms:<10.3f}")
        
        # Synchronization
        print(f"\n### SYNCHRONIZATION (ref: {self.sync_metrics.reference_sensor}) ###")
        print(f"{'Sensor':<15} {'Offset (ms)':<15} {'Std (ms)':<12} {'Drift (ms)':<12}")
        print("-" * 54)
        for sensor_name in self.sync_metrics.offsets_mean_ms.keys():
            offset = self.sync_metrics.offsets_mean_ms[sensor_name]
            std = self.sync_metrics.offsets_std_ms[sensor_name]
            drft = self.sync_metrics.drift_ms[sensor_name]
            print(f"{sensor_name:<15} {offset:<15.2f} {std:<12.3f} {drft:<12.2f}")
        
        print(f"\nMaximum jitter: {self.sync_metrics.max_jitter_ms:.3f} ms")
        
        # Quality assessment
        print("\n### QUALITY ASSESSMENT ###")
        if self.sync_metrics.max_jitter_ms < 2.0:
            print("✓ Excellent synchronization (jitter < 2ms)")
        elif self.sync_metrics.max_jitter_ms < 5.0:
            print("✓ Good synchronization (jitter < 5ms)")
        else:
            print("⚠ Warning: High jitter detected")
        
        if all(missing[s] == 0 for s in missing):
            print("✓ No missing frames")
        else:
            print("⚠ Warning: Missing frames detected")
        
        print("=" * 70)
    
    def export_to_json(self, output_path: str) -> None:
        """
        Export analysis results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        import json
        
        summary = self.get_summary_dict()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Exported timing analysis to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = KITTITimestampAnalyzer(
        "../raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync"
    )
    
    # Run full analysis
    analyzer.load_all_timestamps()
    analyzer.analyze_frame_rates()
    analyzer.analyze_synchronization(reference='lidar')
    
    # Print summary
    analyzer.print_summary()
    
    # Export to JSON
    analyzer.export_to_json("outputs/timestamp_analysis.json")


