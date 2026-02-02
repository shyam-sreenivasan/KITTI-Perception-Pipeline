from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

class KITTICameraLoader:
    """KITTI-specific camera data loader."""
    
    CAMERA_INFO = {
        'image_00': {'name': 'Grayscale Left', 'type': 'grayscale'},
        'image_01': {'name': 'Grayscale Right', 'type': 'grayscale'},
        'image_02': {'name': 'Color Left', 'type': 'color'},
        'image_03': {'name': 'Color Right', 'type': 'color'}
    }
    
    def __init__(self, drive_path: str, camera_id: str = 'image_02'):
        self.drive_path = Path(drive_path)
        self.camera_id = camera_id
        self.camera_path = self.drive_path / camera_id / 'data'
        
        if camera_id not in self.CAMERA_INFO:
            raise ValueError(f"Invalid camera_id: {camera_id}")
        
        if not self.camera_path.exists():
            raise FileNotFoundError(f"Camera data not found: {self.camera_path}")
        
        self.camera_name = self.CAMERA_INFO[camera_id]['name']
        self.camera_type = self.CAMERA_INFO[camera_id]['type']
    
    def load_frame(self, frame_idx: int) -> np.ndarray:
        """Load a single frame as RGB numpy array."""
        img_file = self.camera_path / f'{frame_idx:010d}.png'
        
        if not img_file.exists():
            raise FileNotFoundError(f"Frame {frame_idx} not found")
        
        img_bgr = cv2.imread(str(img_file))
        
        if self.camera_type == 'grayscale':
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img
    
    def load_sequence(self, start_idx: int = 0, 
                     end_idx: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
        """Load a sequence of frames."""
        frame_files = sorted(self.camera_path.glob('*.png'))
        available_indices = [int(f.stem) for f in frame_files]
        
        if end_idx is None:
            end_idx = max(available_indices) + 1
        
        frame_indices = [idx for idx in available_indices if start_idx <= idx < end_idx]
        images = [self.load_frame(idx) for idx in frame_indices]
        
        return images, frame_indices


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    loader = KITTICameraLoader(
        "raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync",
        camera_id='image_02'
    )
    
    img = loader.load_frame(0)
    print(f"Loaded image: {img.shape}")