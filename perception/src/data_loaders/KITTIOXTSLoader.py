import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json

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

if __name__ == "__main__":
    loader = KITTIOXTSLoader("raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync")
    data_seq, indices = loader.load_sequence(0, 108)
    print("GPS_IMU data seq, data_seq", data_seq)