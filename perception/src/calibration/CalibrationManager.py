"""
Calibration parameter management for multi-sensor perception.

Provides:
- Calibration parameter loading and parsing
- Intrinsic/extrinsic parameter extraction
- Transform matrix construction

Does NOT include sensor fusion or projection logic.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
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


class CalibrationManager:
    """
    Manage sensor calibration parameters.
    
    Handles:
    - Loading calibration files
    - Parsing intrinsic/extrinsic parameters
    - Providing clean access to calibration data
    
    Does NOT perform:
    - Projection (handled by SensorFusion)
    - Data loading (handled by sensor-specific loaders)
    - Visualization (handled by analyzers)
    
    Example:
        >>> calib = CalibrationManager('data/2011_09_26')
        >>> calib.load_calibration()
        >>> 
        >>> # Get camera parameters
        >>> cam_params = calib.get_camera_intrinsics('02')
        >>> print(cam_params['P_rect'])  # 3x4 projection matrix
        >>> 
        >>> # Get LiDAR transform
        >>> lidar_transform = calib.get_lidar_to_camera_transform()
        >>> print(lidar_transform['T_4x4'])  # 4x4 homogeneous transform
    """
    
    def __init__(self, date_path: str):
        """
        Initialize calibration manager.
        
        Args:
            date_path: Path to calibration directory (e.g., '2011_09_26')
                      Contains: calib_cam_to_cam.txt, calib_velo_to_cam.txt, etc.
        """
        self.date_path = Path(date_path)
        
        # Calibration file paths
        self.calib_files = {
            'cam_to_cam': self.date_path / 'calib_cam_to_cam.txt',
            'velo_to_cam': self.date_path / 'calib_velo_to_cam.txt',
            'imu_to_velo': self.date_path / 'calib_imu_to_velo.txt'
        }
        
        # Parsed calibration data
        self.calibration: Dict = {}
        
        # Verify files exist
        self._verify_files()
    
    def _verify_files(self):
        """Check that calibration files exist."""
        missing = []
        for name, filepath in self.calib_files.items():
            if not filepath.exists():
                missing.append(f"{name}: {filepath}")
        
        if missing:
            raise FileNotFoundError(
                f"Calibration files not found:\n" + "\n".join(missing)
            )
    
    def _parse_calibration_file(self, filepath: Path) -> Dict:
        """
        Parse a single calibration file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            Dictionary mapping parameter name to numpy array or string
        """
        calib = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip() or ':' not in line:
                    continue
                
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    # Try to parse as numeric array
                    values = np.array([float(x) for x in value.split()])
                    calib[key] = values
                except ValueError:
                    # Keep as string (e.g., calib_time)
                    calib[key] = value
        
        return calib
    
    def load_calibration(self) -> Dict:
        """
        Load and parse all calibration files.
        
        Returns:
            Dictionary with parsed calibration data
        """
        print("="*70)
        print("LOADING CALIBRATION PARAMETERS")
        print("="*70)
        
        for name, filepath in self.calib_files.items():
            self.calibration[name] = self._parse_calibration_file(filepath)
            num_params = len(self.calibration[name])
            print(f"✓ {name:15s}: {num_params:2d} parameters")
        
        print("="*70)
        
        return self.calibration
    
    def get_camera_intrinsics(self, camera_id: str = '02') -> Dict[str, np.ndarray]:
        """
        Get camera intrinsic parameters.
        
        Args:
            camera_id: Camera ID ('00', '01', '02', '03')
            
        Returns:
            Dictionary with:
                - K: Camera matrix (3x3)
                - D: Distortion coefficients (5,)
                - R_rect: Rectification rotation (3x3)
                - P_rect: Projection matrix (3x4)
                - image_size: (width, height)
        """
        if not self.calibration:
            raise ValueError("Calibration not loaded. Call load_calibration() first.")
        
        cam_calib = self.calibration['cam_to_cam']
        
        return {
            'K': cam_calib[f'K_{camera_id}'].reshape(3, 3),
            'D': cam_calib[f'D_{camera_id}'],
            'R_rect': cam_calib[f'R_rect_{camera_id}'].reshape(3, 3),
            'P_rect': cam_calib[f'P_rect_{camera_id}'].reshape(3, 4),
            'image_size': (
                int(cam_calib[f'S_rect_{camera_id}'][0]),
                int(cam_calib[f'S_rect_{camera_id}'][1])
            )
        }
    
    def get_lidar_to_camera_transform(self) -> Dict[str, np.ndarray]:
        """
        Get LiDAR-to-camera extrinsic transform.
        
        Returns:
            Dictionary with:
                - R: Rotation matrix (3x3)
                - T: Translation vector (3x1)
                - T_4x4: Homogeneous transform (4x4)
        """
        if not self.calibration:
            raise ValueError("Calibration not loaded. Call load_calibration() first.")
        
        velo_calib = self.calibration['velo_to_cam']
        
        R = velo_calib['R'].reshape(3, 3)
        T = velo_calib['T'].reshape(3, 1)
        
        # Build 4x4 homogeneous transform
        T_4x4 = np.eye(4)
        T_4x4[:3, :3] = R
        T_4x4[:3, 3] = T.flatten()
        
        return {
            'R': R,
            'T': T,
            'T_4x4': T_4x4
        }
    
    def get_imu_to_lidar_transform(self) -> Dict[str, np.ndarray]:
        """
        Get IMU-to-LiDAR extrinsic transform.
        
        Returns:
            Dictionary with:
                - R: Rotation matrix (3x3)
                - T: Translation vector (3x1)
                - T_4x4: Homogeneous transform (4x4)
        """
        if not self.calibration:
            raise ValueError("Calibration not loaded. Call load_calibration() first.")
        
        imu_calib = self.calibration['imu_to_velo']
        
        R = imu_calib['R'].reshape(3, 3)
        T = imu_calib['T'].reshape(3, 1)
        
        T_4x4 = np.eye(4)
        T_4x4[:3, :3] = R
        T_4x4[:3, 3] = T.flatten()
        
        return {
            'R': R,
            'T': T,
            'T_4x4': T_4x4
        }
    
    def get_all_transforms(self) -> Dict[str, Dict]:
        """
        Get all transformation matrices.
        
        Returns:
            Dictionary with all available transforms
        """
        return {
            'lidar_to_camera': self.get_lidar_to_camera_transform(),
            'imu_to_lidar': self.get_imu_to_lidar_transform()
        }
    
    def export_calibration(self, output_path: str) -> None:
        """
        Export calibration data to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if not self.calibration:
            raise ValueError("Calibration not loaded. Call load_calibration() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        clean_calib = convert_numpy_types(self.calibration)
        
        with open(output_path, 'w') as f:
            json.dump(clean_calib, f, indent=2)
        
        print(f"✓ Exported calibration to {output_path}")
    
    def print_summary(self):
        """Print human-readable summary of calibration parameters."""
        if not self.calibration:
            raise ValueError("Calibration not loaded. Call load_calibration() first.")
        
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY")
        print("="*70)
        
        # Camera parameters
        print("\n### CAMERA PARAMETERS (image_02 - Color Left) ###")
        cam_params = self.get_camera_intrinsics('02')
        print(f"Image size: {cam_params['image_size']}")
        print(f"Focal length (fx, fy): ({cam_params['P_rect'][0, 0]:.2f}, "
              f"{cam_params['P_rect'][1, 1]:.2f}) pixels")
        print(f"Principal point (cx, cy): ({cam_params['P_rect'][0, 2]:.2f}, "
              f"{cam_params['P_rect'][1, 2]:.2f}) pixels")
        
        # LiDAR-Camera transform
        print("\n### LIDAR-TO-CAMERA TRANSFORM ###")
        lidar_cam = self.get_lidar_to_camera_transform()
        print(f"Translation: {lidar_cam['T'].flatten()}")
        print(f"Translation magnitude: {np.linalg.norm(lidar_cam['T']):.3f} m")
        
        # IMU-LiDAR transform
        print("\n### IMU-TO-LIDAR TRANSFORM ###")
        imu_lidar = self.get_imu_to_lidar_transform()
        print(f"Translation: {imu_lidar['T'].flatten()}")
        print(f"Translation magnitude: {np.linalg.norm(imu_lidar['T']):.3f} m")
        
        print("="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize calibration manager
    BASE_PATH = Path("raw_data_downloader/2011_09_26")
    
    print("="*70)
    print("CALIBRATION MANAGER - STANDALONE DEMO")
    print("="*70)
    
    # Load calibration
    calib_mgr = CalibrationManager(BASE_PATH)
    calib_mgr.load_calibration()
    
    # Print summary
    calib_mgr.print_summary()
    
    # Export to JSON
    calib_mgr.export_calibration("outputs/calibration.json")
    
    # Access specific parameters
    print("\n### ACCESSING SPECIFIC PARAMETERS ###")
    cam_intrinsics = calib_mgr.get_camera_intrinsics('02')
    print(f"\nCamera 02 parameters:")
    print(f"  Image size: {cam_intrinsics['image_size']}")
    print(f"  Focal length (fx): {cam_intrinsics['P_rect'][0, 0]:.2f} pixels")
    print(f"  Principal point (cx, cy): ({cam_intrinsics['P_rect'][0, 2]:.1f}, "
          f"{cam_intrinsics['P_rect'][1, 2]:.1f}) pixels")
    
    lidar_transform = calib_mgr.get_lidar_to_camera_transform()
    print(f"\nLiDAR-Camera transform:")
    print(f"  Rotation matrix R:\n{lidar_transform['R']}")
    print(f"  Translation T: {lidar_transform['T'].flatten()}")
    print(f"  Translation magnitude: {np.linalg.norm(lidar_transform['T']):.3f} m")
    
    print("\n" + "="*70)
    print("✓ CALIBRATION MANAGER READY")
    print("="*70)
    print("\nThis module can now be imported by:")
    print("  - SensorFusion (for projection)")
    print("  - Any other module needing calibration parameters")