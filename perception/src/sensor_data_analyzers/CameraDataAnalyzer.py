"""
Generic camera sensor data quality analyzer with visualization support.

Hardware-agnostic analyzer that works with any camera image data.
Provides comprehensive quality metrics for perception systems.
"""

import numpy as np
import cv2
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
class CameraFrameMetrics:
    """Metrics for a single camera frame."""
    frame_idx: int
    timestamp: float
    resolution: Tuple[int, int]
    channels: int
    brightness_stats: Dict[str, float]
    contrast: float
    dynamic_range: float
    overexposed_pct: float
    underexposed_pct: float
    sharpness: float
    edge_density: float
    edge_pixel_count: int
    harris_corners: int
    orb_keypoints: int
    fast_keypoints: int
    shi_tomasi_corners: int
    feature_distribution: Dict[str, int]
    quality_flags: Dict[str, bool]


@dataclass
class CameraSequenceMetrics:
    """Aggregate metrics for multiple frames."""
    num_frames: int
    frame_indices: List[int]
    brightness_mean: float
    brightness_std: float
    brightness_variation_pct: float
    sharpness_mean: float
    sharpness_std: float
    sharpness_variation_pct: float
    exposure_quality_mean: float
    orb_keypoints_mean: float
    orb_keypoints_std: float
    quality_flags: Dict[str, bool]


class CameraDataAnalyzer:
    """
    Generic camera image quality analyzer.
    
    Hardware-agnostic analyzer that accepts image data directly.
    Suitable for any camera: monocular, stereo, color, grayscale.
    
    Example:
        >>> analyzer = CameraDataAnalyzer()
        >>> image = cv2.imread('frame.png')
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> metrics = analyzer.analyze_frame(image, frame_idx=0)
        >>> fig = analyzer.visualize_frame(image, frame_idx=0)
        >>> plt.show()
    """
    
    # Quality thresholds (configurable)
    DEFAULT_THRESHOLDS = {
        'overexposed_max': 5.0,
        'underexposed_max': 10.0,
        'contrast_min': 30.0,
        'sharpness_min': 100.0,
        'edge_density_min': 2.0,
        'orb_keypoints_min': 500
    }
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """Initialize camera analyzer."""
        self.thresholds = thresholds if thresholds else self.DEFAULT_THRESHOLDS.copy()
    
    def _compute_luminance(self, img: np.ndarray) -> np.ndarray:
        """Compute luminance from color or grayscale image."""
        if len(img.shape) == 3:
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            luminance = img.astype(float)
        return luminance
    
    def _compute_brightness_stats(self, img: np.ndarray) -> Dict[str, float]:
        """Calculate brightness statistics."""
        luminance = self._compute_luminance(img)
        return {
            'min': float(luminance.min()),
            'max': float(luminance.max()),
            'mean': float(luminance.mean()),
            'std': float(luminance.std()),
            'median': float(np.median(luminance))
        }
    
    def _compute_exposure_quality(self, img: np.ndarray) -> Tuple[float, float]:
        """Assess exposure quality."""
        luminance = self._compute_luminance(img)
        total_pixels = luminance.size
        
        overexposed = np.sum(luminance > 240)
        underexposed = np.sum(luminance < 15)
        
        overexposed_pct = float((overexposed / total_pixels) * 100)
        underexposed_pct = float((underexposed / total_pixels) * 100)
        
        return overexposed_pct, underexposed_pct
    
    def _compute_sharpness(self, img: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        
        return sharpness
    
    def _compute_edge_density(self, img: np.ndarray) -> Tuple[float, int]:
        """Compute edge density using Canny edge detector."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = int(np.sum(edges > 0))
        total_pixels = gray.size
        edge_density = float((edge_pixels / total_pixels) * 100)
        
        return edge_density, edge_pixels
    
    def _detect_features(self, img: np.ndarray) -> Dict[str, any]:
        """Run multiple feature detectors."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Harris corners
        harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        harris_threshold = 0.01 * harris_corners.max() if harris_corners.max() > 0 else 0
        num_harris = int(np.sum(harris_corners > harris_threshold))
        
        # ORB keypoints
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints_orb, _ = orb.detectAndCompute(gray, None)
        num_orb = len(keypoints_orb)
        
        # FAST corners
        fast = cv2.FastFeatureDetector_create(threshold=20)
        keypoints_fast = fast.detect(gray, None)
        num_fast = len(keypoints_fast)
        
        # Shi-Tomasi corners
        corners_shi_tomasi = cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=10
        )
        num_shi_tomasi = len(corners_shi_tomasi) if corners_shi_tomasi is not None else 0
        
        return {
            'harris': num_harris,
            'orb': num_orb,
            'fast': num_fast,
            'shi_tomasi': num_shi_tomasi,
            'orb_keypoints_obj': keypoints_orb,
            'gray': gray,
            'edges': cv2.Canny(gray, 50, 150)
        }
    
    def _compute_feature_distribution(self, keypoints, img_height: int) -> Dict[str, int]:
        """Compute feature distribution across vertical regions."""
        strip_height = img_height // 5
        regions = ['top', 'upper_mid', 'middle', 'lower_mid', 'bottom']
        
        distribution = {}
        for i, region_name in enumerate(regions):
            y_start = i * strip_height
            y_end = (i + 1) * strip_height if i < 4 else img_height
            count = sum(1 for kp in keypoints if y_start <= kp.pt[1] < y_end)
            distribution[region_name] = count
        
        return distribution
    
    def _evaluate_quality_flags(self, metrics: Dict) -> Dict[str, bool]:
        """Evaluate quality flags based on thresholds."""
        return {
            'good_exposure': bool(
                metrics['overexposed_pct'] < self.thresholds['overexposed_max'] and
                metrics['underexposed_pct'] < self.thresholds['underexposed_max']
            ),
            'sufficient_contrast': bool(metrics['contrast'] >= self.thresholds['contrast_min']),
            'sharp_image': bool(metrics['sharpness'] >= self.thresholds['sharpness_min']),
            'sufficient_edges': bool(metrics['edge_density'] >= self.thresholds['edge_density_min']),
            'sufficient_features': bool(metrics['orb_keypoints'] >= self.thresholds['orb_keypoints_min'])
        }
    
    def analyze_frame(self, img: np.ndarray, frame_idx: int = 0, 
                     timestamp: float = 0.0) -> Dict:
        """
        Comprehensive analysis of a single camera frame.
        
        Args:
            img: Image array (H, W) for grayscale or (H, W, 3) for color (RGB)
            frame_idx: Frame identifier
            timestamp: Frame timestamp in seconds
            
        Returns:
            Dictionary with frame metrics (JSON-serializable)
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be numpy array")
        
        if len(img.shape) not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
        
        # Compute all metrics
        brightness_stats = self._compute_brightness_stats(img)
        luminance = self._compute_luminance(img)
        contrast = float(luminance.std())
        dynamic_range = float(brightness_stats['max'] - brightness_stats['min'])
        
        overexposed_pct, underexposed_pct = self._compute_exposure_quality(img)
        sharpness = self._compute_sharpness(img)
        edge_density, edge_pixel_count = self._compute_edge_density(img)
        
        features = self._detect_features(img)
        feature_distribution = self._compute_feature_distribution(
            features['orb_keypoints_obj'], img.shape[0]
        )
        
        # Build metrics dict
        metrics_dict = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'resolution': (int(img.shape[1]), int(img.shape[0])),
            'channels': int(img.shape[2] if len(img.shape) == 3 else 1),
            'brightness_stats': brightness_stats,
            'contrast': contrast,
            'dynamic_range': dynamic_range,
            'overexposed_pct': overexposed_pct,
            'underexposed_pct': underexposed_pct,
            'sharpness': sharpness,
            'edge_density': edge_density,
            'edge_pixel_count': edge_pixel_count,
            'harris_corners': features['harris'],
            'orb_keypoints': features['orb'],
            'fast_keypoints': features['fast'],
            'shi_tomasi_corners': features['shi_tomasi'],
            'feature_distribution': feature_distribution
        }
        
        metrics_dict['quality_flags'] = self._evaluate_quality_flags(metrics_dict)
        
        return metrics_dict
    
    def visualize_frame(self, img: np.ndarray, frame_idx: int = 0,
                       save_path: Optional[str] = None, mode: str = 'basic') -> plt.Figure:
        """
        Create visualization of camera frame.
        
        Args:
            img: Image array (H, W) or (H, W, 3) in RGB order
            frame_idx: Frame identifier for title
            save_path: Optional path to save figure
            mode: Visualization mode
                  'basic' - 4-panel simple view (default)
                  'comprehensive' - 9-panel detailed feature analysis
            
        Returns:
            Matplotlib Figure object
        """
        if mode == 'comprehensive':
            return self._visualize_comprehensive(img, frame_idx, save_path)
        else:
            return self._visualize_basic(img, frame_idx, save_path)
    
    def _visualize_basic(self, img: np.ndarray, frame_idx: int = 0,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 4-panel basic visualization.
        
        Panels:
        1. Raw Image
        2. Feature Detection (ORB keypoints)
        3. Brightness Histogram
        4. Edge Detection (Canny)
        """
        # Detect features (includes gray and edges)
        features = self._detect_features(img)
        gray = features['gray']
        edges = features['edges']
        keypoints_orb = features['orb_keypoints_obj']
        
        # Compute luminance for histogram
        luminance = self._compute_luminance(img)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # ========== PANEL 1: Raw Image ==========
        ax1 = fig.add_subplot(2, 2, 1)
        if len(img.shape) == 3:
            ax1.imshow(img)
        else:
            ax1.imshow(img, cmap='gray')
        ax1.set_title('Raw Image', fontweight='bold', fontsize=12)
        ax1.axis('off')
        
        # ========== PANEL 2: Feature Detection ==========
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Draw ORB keypoints on image
        if len(img.shape) == 3:
            img_with_features = img.copy()
            # Convert to BGR for cv2.drawKeypoints, then back to RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_with_kp = cv2.drawKeypoints(img_bgr, keypoints_orb, None, 
                                           color=(0, 255, 0), 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_with_features = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
        else:
            img_with_features = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            img_with_features = cv2.drawKeypoints(gray, keypoints_orb, None,
                                                  color=(0, 255, 0),
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        ax2.imshow(img_with_features)
        ax2.set_title(f'Feature Detection ({len(keypoints_orb):,} ORB keypoints)', 
                     fontweight='bold', fontsize=12)
        ax2.axis('off')
        
        # ========== PANEL 3: Brightness Histogram ==========
        ax3 = fig.add_subplot(2, 2, 3)
        
        hist, bins = np.histogram(luminance.flatten(), bins=50, range=(0, 255))
        ax3.bar(bins[:-1], hist, width=np.diff(bins), color='steelblue', 
               edgecolor='black', alpha=0.7)
        ax3.axvline(luminance.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {luminance.mean():.1f}')
        ax3.axvline(15, color='orange', linestyle=':', linewidth=2, alpha=0.5,
                   label='Underexposed (<15)')
        ax3.axvline(240, color='orange', linestyle=':', linewidth=2, alpha=0.5,
                   label='Overexposed (>240)')
        
        ax3.set_xlabel('Brightness Value', fontsize=10)
        ax3.set_ylabel('Pixel Count', fontsize=10)
        ax3.set_title('Brightness/Exposure Histogram', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ========== PANEL 4: Edge Detection ==========
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(edges, cmap='gray')
        
        edge_pct = (np.sum(edges > 0) / edges.size) * 100
        ax4.set_title(f'Edge Detection ({edge_pct:.2f}% edge pixels)', 
                     fontweight='bold', fontsize=12)
        ax4.axis('off')
        
        # Overall title with quality summary
        metrics = self.analyze_frame(img, frame_idx)
        quality_str = "✓ GOOD" if all(metrics['quality_flags'].values()) else "⚠ CHECK"
        
        fig.suptitle(f'Camera Frame {frame_idx} - Quality Visualization {quality_str}\n'
                    f'Resolution: {img.shape[1]}×{img.shape[0]} | '
                    f'Brightness: {luminance.mean():.1f} | '
                    f'Sharpness: {metrics["sharpness"]:.1f}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        return fig
    
    def _visualize_comprehensive(self, img: np.ndarray, frame_idx: int = 0,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 9-panel comprehensive feature analysis visualization.
        
        Panels:
        1. Original Image
        2. Harris Corners
        3. ORB Keypoints
        4. FAST Keypoints
        5. Canny Edges
        6. Shi-Tomasi Corners
        7. Feature Density Heatmap
        8. Vertical Distribution
        9. Method Comparison
        """
        # Detect all features
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_display = img.copy()
        else:
            gray = img
            img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Harris corners
        harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        harris_threshold = 0.01 * harris_corners.max() if harris_corners.max() > 0 else 0
        num_harris = int(np.sum(harris_corners > harris_threshold))
        
        # ORB
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints_orb, _ = orb.detectAndCompute(gray, None)
        
        # FAST
        fast = cv2.FastFeatureDetector_create(threshold=20)
        keypoints_fast = fast.detect(gray, None)
        
        # Shi-Tomasi
        corners_shi_tomasi = cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=10
        )
        num_shi_tomasi = len(corners_shi_tomasi) if corners_shi_tomasi is not None else 0
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = int(np.sum(edges > 0))
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        
        # ========== PANEL 1: Original Image ==========
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.imshow(img_display)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # ========== PANEL 2: Harris Corners ==========
        ax2 = fig.add_subplot(3, 3, 2)
        img_harris = img_display.copy()
        img_harris[harris_corners > harris_threshold] = [255, 0, 0]
        ax2.imshow(img_harris)
        ax2.set_title(f'Harris Corners ({num_harris:,} detected)', fontsize=12)
        ax2.axis('off')
        
        # ========== PANEL 3: ORB Keypoints ==========
        ax3 = fig.add_subplot(3, 3, 3)
        img_orb = cv2.drawKeypoints(gray, keypoints_orb, None,
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax3.imshow(img_orb, cmap='gray')
        ax3.set_title(f'ORB Keypoints ({len(keypoints_orb):,} detected)', fontsize=12)
        ax3.axis('off')
        
        # ========== PANEL 4: FAST Keypoints ==========
        ax4 = fig.add_subplot(3, 3, 4)
        img_fast = cv2.drawKeypoints(gray, keypoints_fast, None, color=(255, 0, 0))
        ax4.imshow(img_fast, cmap='gray')
        ax4.set_title(f'FAST Keypoints ({len(keypoints_fast):,} detected)', fontsize=12)
        ax4.axis('off')
        
        # ========== PANEL 5: Canny Edges ==========
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.imshow(edges, cmap='gray')
        ax5.set_title(f'Canny Edges ({edge_pixels:,} pixels)', fontsize=12)
        ax5.axis('off')
        
        # ========== PANEL 6: Shi-Tomasi Corners ==========
        ax6 = fig.add_subplot(3, 3, 6)
        img_shi_tomasi = img_display.copy()
        if corners_shi_tomasi is not None:
            for corner in corners_shi_tomasi:
                x, y = corner.ravel()
                cv2.circle(img_shi_tomasi, (int(x), int(y)), 3, (0, 255, 0), -1)
        ax6.imshow(img_shi_tomasi)
        ax6.set_title(f'Shi-Tomasi Corners ({num_shi_tomasi:,} detected)', fontsize=12)
        ax6.axis('off')
        
        # ========== PANEL 7: Feature Density Heatmap ==========
        ax7 = fig.add_subplot(3, 3, 7)
        if len(keypoints_orb) > 0:
            kp_coords = np.array([kp.pt for kp in keypoints_orb])
            heatmap, xedges, yedges = np.histogram2d(kp_coords[:, 0], kp_coords[:, 1],
                                                     bins=[50, 15])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax7.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
            ax7.set_title('ORB Feature Density Heatmap', fontsize=12)
            ax7.set_xlabel('Image Width (pixels)', fontsize=10)
            ax7.set_ylabel('Image Height (pixels)', fontsize=10)
            plt.colorbar(im, ax=ax7, label='Features per region', shrink=0.8)
        else:
            ax7.text(0.5, 0.5, 'No features detected', ha='center', va='center',
                    transform=ax7.transAxes)
            ax7.set_title('ORB Feature Density Heatmap', fontsize=12)
        
        # ========== PANEL 8: Feature Distribution by Region ==========
        ax8 = fig.add_subplot(3, 3, 8)
        strip_height = gray.shape[0] // 5
        strip_labels = ['Top', 'Upper-Mid', 'Middle', 'Lower-Mid', 'Bottom']
        strip_counts = []
        
        for i in range(5):
            y_start = i * strip_height
            y_end = (i + 1) * strip_height if i < 4 else gray.shape[0]
            count = sum(1 for kp in keypoints_orb if y_start <= kp.pt[1] < y_end)
            strip_counts.append(count)
        
        bars = ax8.bar(strip_labels, strip_counts, color='steelblue', edgecolor='black')
        ax8.set_ylabel('Number of ORB Features', fontsize=10)
        ax8.set_title('Feature Distribution by Vertical Region', fontsize=12)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total_features = sum(strip_counts)
        if total_features > 0:
            for bar, count in zip(bars, strip_counts):
                height = bar.get_height()
                pct = (count / total_features) * 100
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # ========== PANEL 9: Method Comparison ==========
        ax9 = fig.add_subplot(3, 3, 9)
        methods = ['Harris', 'ORB', 'FAST', 'Canny\nEdges', 'Shi-Tomasi']
        counts = [num_harris, len(keypoints_orb), len(keypoints_fast),
                 edge_pixels, num_shi_tomasi]
        
        bars = ax9.bar(methods, counts, color=['red', 'green', 'blue', 'orange', 'purple'])
        ax9.set_ylabel('Feature Count', fontsize=10)
        ax9.set_title('Feature Detection Method Comparison', fontsize=12)
        ax9.set_yscale('log')  # Log scale because Canny has many more
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Overall title
        fig.suptitle(f'Camera Frame {frame_idx} - Comprehensive Feature Analysis',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        return fig
    
    def analyze_sequence(self, images: List[np.ndarray], 
                        frame_indices: Optional[List[int]] = None,
                        timestamps: Optional[List[float]] = None) -> Dict:
        """Analyze a sequence of camera frames."""
        if not images:
            raise ValueError("Image list cannot be empty")
        
        n = len(images)
        if frame_indices is None:
            frame_indices = list(range(n))
        if timestamps is None:
            timestamps = [0.0] * n
        
        # Analyze each frame
        frame_metrics = []
        for img, idx, ts in zip(images, frame_indices, timestamps):
            metrics = self.analyze_frame(img, idx, ts)
            frame_metrics.append(metrics)
        
        # Aggregate statistics
        brightness_vals = [m['brightness_stats']['mean'] for m in frame_metrics]
        sharpness_vals = [m['sharpness'] for m in frame_metrics]
        orb_vals = [m['orb_keypoints'] for m in frame_metrics]
        
        exposure_quality = [
            100 - m['overexposed_pct'] - m['underexposed_pct']
            for m in frame_metrics
        ]
        
        brightness_mean = float(np.mean(brightness_vals))
        brightness_std = float(np.std(brightness_vals))
        brightness_variation = (brightness_std / brightness_mean * 100) if brightness_mean > 0 else 0
        
        sharpness_mean = float(np.mean(sharpness_vals))
        sharpness_std = float(np.std(sharpness_vals))
        sharpness_variation = (sharpness_std / sharpness_mean * 100) if sharpness_mean > 0 else 0
        
        quality_flags = {
            'consistent_brightness': bool(brightness_variation < 10.0),
            'consistent_sharpness': bool(sharpness_variation < 20.0),
            'good_exposure_overall': bool(np.mean(exposure_quality) > 80.0),
            'sufficient_features': bool(np.mean(orb_vals) >= self.thresholds['orb_keypoints_min'])
        }
        
        summary = CameraSequenceMetrics(
            num_frames=n,
            frame_indices=frame_indices,
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            brightness_variation_pct=float(brightness_variation),
            sharpness_mean=sharpness_mean,
            sharpness_std=sharpness_std,
            sharpness_variation_pct=float(sharpness_variation),
            exposure_quality_mean=float(np.mean(exposure_quality)),
            orb_keypoints_mean=float(np.mean(orb_vals)),
            orb_keypoints_std=float(np.std(orb_vals)),
            quality_flags=quality_flags
        )
        
        return {
            'analysis_type': 'sequence',
            'summary': asdict(summary),
            'frames': frame_metrics
        }
    
    def export_analysis(self, output_path: str, analysis_result: Dict) -> None:
        """Export analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        clean_result = convert_numpy_types(analysis_result)
        with open(output_path, 'w') as f:
            json.dump(clean_result, f, indent=2)
        
        print(f"✓ Exported camera analysis to {output_path}")


# ============================================================================
# KITTI LOADER (Usage Example)
# ============================================================================

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
    # Initialize analyzer and loader
    analyzer = CameraDataAnalyzer()
    try:
        from ..data_loaders import KITTICameraLoader
    except e:
        import sys
        print(e)
        sys.exit(0)

    loader = KITTICameraLoader(
        "raw_data_downloader/2011_09_26/2011_09_26_drive_0001_sync",
        camera_id='image_02'
    )
    
    print("="*70)
    print(f"CAMERA DATA QUALITY ANALYSIS: {loader.camera_name}")
    print("="*70)
    
    # Single frame analysis with visualization
    print("\n### ANALYZING FRAME 0 ###")
    img = loader.load_frame(0)
    print(f"Loaded image: {img.shape}")
    
    # Compute metrics
    metrics = analyzer.analyze_frame(img, frame_idx=0, timestamp=0.0)
    print(f"Resolution: {metrics['resolution']}")
    print(f"Brightness: {metrics['brightness_stats']['mean']:.2f}")
    print(f"Sharpness: {metrics['sharpness']:.2f}")
    print(f"ORB features: {metrics['orb_keypoints']}")
    print(f"Quality flags: {metrics['quality_flags']}")
    
    # Create visualization
    fig = analyzer.visualize_frame(img, frame_idx=0,
                                   save_path="outputs/camera_frame_0_viz.png")
    
    # Or comprehensive mode with all feature detectors
    fig_comp = analyzer.visualize_frame(img, frame_idx=0, mode='comprehensive',
                                       save_path="outputs/camera_frame_0_comprehensive.png")
    
    # Export metrics
    analyzer.export_analysis("outputs/camera_frame_0.json", metrics)
    
    print("\n✓ Analysis complete!")
    print("  - Metrics saved to: outputs/camera_frame_0.json")
    print("  - Basic viz saved to: outputs/camera_frame_0_viz.png")
    print("  - Comprehensive viz saved to: outputs/camera_frame_0_comprehensive.png")