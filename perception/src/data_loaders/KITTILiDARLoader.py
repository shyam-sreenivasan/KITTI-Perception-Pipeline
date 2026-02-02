# ============================================================================
# KITTI LOADER (Usage Example)
# ============================================================================
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple

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


if __name__ == "__main__":
    drive_path = "raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync"
    data_loader = KITTILiDARLoader(drive_path)
    points = data_loader.load_frame(0)
    print("Frame 0", points)

    print("---------------------------------")
    points_list, frame_indices = data_loader.load_sequence(0)
    print("Sequence", points, frame_indices)