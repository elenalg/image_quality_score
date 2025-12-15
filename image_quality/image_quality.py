#!/usr/bin/env python3
"""
Image Quality - Class to analyze quality of individual images

This module provides a class to evaluate image quality metrics including:
- Sharpness (Laplacian variance)
- Exposure (histogram analysis)
- BRISQUE (via IQA-PyTorch, optional) - evaluates technical distortions

The class can work with numpy arrays directly or with image paths.
It provides a unified quality score (0-1) combining all metrics.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Union, Any
import json

# Try to import IQA metrics (optional dependency)
try:
    from .iqa_metrics import IQAMetrics, is_iqa_available
    IQA_AVAILABLE = True
except ImportError:
    IQA_AVAILABLE = False
    IQAMetrics = None

# Configure logging (only get logger, don't configure basicConfig here)
logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_FILE = Path(__file__).parent / 'recommended_config.json'
# Default rounding for reported scores/metrics
DEFAULT_ROUND_DIGITS = 4


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load recommended configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses default recommended_config.json.
    
    Returns:
        Dictionary with configuration:
        {
            'sharpness_threshold': float,
            'exposure_threshold': float,
            'max_sharpness': float,
            'max_brisque': float,
            'quality_score_thresholds': {
                'conservative': float,
                'moderate': float,
                'strict': float
            }
        }
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        'sharpness_threshold': 100.0,
        'exposure_threshold': 0.1,
        'max_sharpness': 500.0,
        'max_brisque': 100.0,
        'quality_score_thresholds': {
            'conservative': 0.5,
            'moderate': 0.7,
            'strict': 0.85
        }
    }


class ImageQuality:
    """
    Class to analyze quality of individual images.
    
    Can evaluate images from numpy arrays or file paths.
    Provides metrics for sharpness and exposure, and optionally BRISQUE
    (via IQA-PyTorch) for a comprehensive quality score (0-1).
    
    The quality score combines multiple metrics to provide a single value
    indicating overall image quality, where 1.0 = best quality, 0.0 = worst quality.
    """
    
    def __init__(self,
                 sharpness_threshold: Optional[float] = None,
                 exposure_threshold: Optional[float] = None,
                 use_iqa: bool = True,
                 iqa_device: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the image quality analyzer.
        
        Args:
            sharpness_threshold: Minimum sharpness threshold (Laplacian variance). If None, uses recommended config.
            exposure_threshold: Normalized exposure threshold (0-1). If None, uses recommended config.
            use_iqa: Whether to use IQA-PyTorch metrics (BRISQUE) if available.
            iqa_device: Device for IQA metrics ('cpu' or 'cuda'). If None, auto-detects.
            config: Optional configuration dict; defaults to loading recommended_config.json.
        """
        self.config = config or load_config()
        self.sharpness_threshold = sharpness_threshold if sharpness_threshold is not None else self.config.get('sharpness_threshold', 100.0)
        self.exposure_threshold = exposure_threshold if exposure_threshold is not None else self.config.get('exposure_threshold', 0.1)
        self.max_sharpness_default = self.config.get('max_sharpness', 1500.0)
        self.max_brisque_default = self.config.get('max_brisque', 100.0)
        self.use_iqa = use_iqa and IQA_AVAILABLE
        
        # Initialize IQA metrics if requested and available
        self.iqa_metrics = None
        if self.use_iqa:
            try:
                self.iqa_metrics = IQAMetrics(device=iqa_device)
                logger.info("IQA metrics (BRISQUE) enabled")
            except Exception as e:
                logger.warning(f"Could not initialize IQA metrics: {e}. Continuing without IQA.")
                self.use_iqa = False

    @staticmethod
    def _to_float(value: Any, decimals: Optional[int] = None) -> float:
        """Convert numpy/builtin numerics to float, optionally rounding."""
        val = float(value)
        return round(val, decimals) if decimals is not None else val
    
    def load_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Load image from path or return numpy array if already loaded.
        
        Args:
            image_input: Path to image file or numpy array
            
        Returns:
            Image as numpy array (BGR format)
            
        Raises:
            ValueError: If image cannot be loaded
        """
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise ValueError(f"Image file does not exist: {image_path}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            return image
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def calculate_sharpness(self, image: np.ndarray, use_regional: bool = True, use_max: bool = True) -> float:
        """
        Calculate sharpness using the Laplacian variance.
        
        Uses regional analysis (3x3 grid, 9 regions) by default to better handle
        images with focused foreground and blurred background.
        
        Args:
            image: Image as numpy array (BGR or grayscale)
            use_regional: If True, uses 3x3 grid analysis (default: True)
                         If False, uses global variance (legacy method)
            use_max: If True, uses maximum sharpness from regions (passes if at least one region is focused)
                    If False, uses percentile (default: True)
            
        Returns:
            Sharpness value (Laplacian variance)
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if use_regional:
            return self._calculate_sharpness_regional(gray, use_max=use_max)
        else:
            # Legacy global method
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()
    
    def _calculate_sharpness_regional(self, gray: np.ndarray, use_max: bool = True) -> float:
        """
        Calculate sharpness using 3x3 grid (9 regions).
        Uses maximum by default to pass images with at least one focused region.
        
        Args:
            gray: Grayscale image as numpy array
            use_max: If True, returns maximum sharpness (passes if at least one region is focused)
                    If False, returns percentile 75 (more strict)
            
        Returns:
            Sharpness value based on regional analysis
        """
        h, w = gray.shape
        sharpness_values = []
        
        # Divide into 3x3 grid (9 regions)
        region_h = h // 3
        region_w = w // 3
        
        for i in range(3):
            for j in range(3):
                y_start = i * region_h
                y_end = (i + 1) * region_h if i < 2 else h
                x_start = j * region_w
                x_end = (j + 1) * region_w if j < 2 else w
                
                region = gray[y_start:y_end, x_start:x_end]
                
                if region.size > 0:
                    laplacian = cv2.Laplacian(region, cv2.CV_64F)
                    sharpness_values.append(laplacian.var())
        
        if not sharpness_values:
            return 0.0
        
        # Use maximum to pass images with at least one focused region
        if use_max:
            return float(max(sharpness_values))
        else:
            # Use percentile 75 for stricter filtering
            return float(np.percentile(sharpness_values, 75))
    
    def calculate_exposure(self, image: np.ndarray) -> float:
        """
        Calculate normalized exposure metric.
        
        Args:
            image: Image as numpy array (BGR or grayscale)
            
        Returns:
            Normalized exposure value (0-1)
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Normalize histogram
        hist_norm = hist.flatten() / hist.sum()
        
        # Calculate exposure metric (weighted average)
        exposure = np.sum(hist_norm * np.arange(256)) / 255.0
        
        return exposure
    
    def analyze_quality(self, image_input: Union[str, Path, np.ndarray]) -> Dict:
        """
        Analyze quality of an image.
        
        Args:
            image_input: Path to image file or numpy array
            
        Returns:
            Dictionary with quality metrics and analysis results:
            {
                'image_path': Optional[str],
                'sharpness': float,
                'exposure': float,
                'passes_sharpness': bool,
                'passes_exposure': bool,
                'overall_passes': bool,
                'rejection_reason': Optional[str]
            }
        """
        # Load image
        image = self.load_image(image_input)
        image_path = str(image_input) if isinstance(image_input, (str, Path)) else None
        
        # Calculate metrics
        sharpness = self.calculate_sharpness(image)
        exposure = self.calculate_exposure(image)
        
        # Evaluate quality
        passes_sharpness = sharpness >= self.sharpness_threshold
        passes_exposure = self.exposure_threshold <= exposure <= (1 - self.exposure_threshold)
        
        # Determine overall result
        overall_passes = passes_sharpness and passes_exposure
        
        # Determine rejection reason if any
        rejection_reason = None
        if not passes_sharpness:
            rejection_reason = f"low_sharpness_{sharpness:.1f}"
        elif not passes_exposure:
            rejection_reason = f"bad_exposure_{exposure:.3f}"
        
        # Build result dictionary (cast to Python floats for consistency)
        result = {
            'image_path': image_path,
            'sharpness': self._to_float(sharpness, DEFAULT_ROUND_DIGITS),
            'exposure': self._to_float(exposure, DEFAULT_ROUND_DIGITS),
            'passes_sharpness': passes_sharpness,
            'passes_exposure': passes_exposure,
            'overall_passes': overall_passes,
            'rejection_reason': rejection_reason
        }
        
        return result
    
    def analyze_batch(self, image_paths: list) -> Dict:
        """
        Analyze quality of multiple images.
        
        Args:
            image_paths: List of image paths or numpy arrays
            
        Returns:
            Dictionary with batch results:
            {
                'results': list of individual analysis results,
                'summary': {
                    'total': int,
                    'passed': int,
                    'failed': int,
                    'errors': int,
                    'pass_rate': float,
                    'rejection_stats': dict
                }
            }
        """
        results = []
        rejection_stats = {
            'low_sharpness': 0,
            'bad_exposure': 0
        }
        # Count images that failed to load/process (distinct from quality failures)
        error_count = 0
        
        for image_input in image_paths:
            try:
                result = self.analyze_quality(image_input)
                results.append(result)
                
                # Update statistics
                if result['rejection_reason']:
                    if result['rejection_reason'].startswith('low_sharpness'):
                        rejection_stats['low_sharpness'] += 1
                    elif result['rejection_reason'].startswith('bad_exposure'):
                        rejection_stats['bad_exposure'] += 1
            except Exception as e:
                logger.error(f"Error analyzing image {image_input}: {e}")
                error_count += 1
                results.append({
                    'image_path': str(image_input),
                    'error': str(e),
                    'overall_passes': False,
                    'rejection_reason': 'error_loading'
                })
        
        # Calculate summary
        total = len(results)
        passed = sum(1 for r in results if r.get('overall_passes', False))
        failed = total - passed
        
        summary = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': error_count,  # Operational errors (I/O, decoding), not quality rejections
            'pass_rate': passed / total if total > 0 else 0.0,
            'rejection_stats': rejection_stats
        }
        
        return {
            'results': results,
            'summary': summary
        }
    
    def analyze_directory(self, directory_path: Union[str, Path],
                          extensions: Optional[list] = None,
                          recursive: bool = False) -> Dict:
        """
        Analyze quality of all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process (e.g., ['.jpg', '.png']).
                       If None, uses default: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            recursive: Whether to search recursively in subdirectories
            
        Returns:
            Dictionary with batch results (same format as analyze_batch):
            {
                'results': list of individual analysis results,
                'summary': {
                    'total': int,
                    'passed': int,
                    'failed': int,
                    'errors': int,
                    'pass_rate': float,
                    'rejection_stats': dict
                }
            }
            
        Raises:
            ValueError: If directory does not exist
        """
        image_paths = self.collect_image_paths(directory_path, extensions=extensions, recursive=recursive)
        
        logger.info(f"Found {len(image_paths)} image(s) in directory: {directory_path}")
        
        # Use analyze_batch to process all images
        return self.analyze_batch(image_paths)
    
    @staticmethod
    def collect_image_paths(directory_path: Union[str, Path],
                            extensions: Optional[list] = None,
                            recursive: bool = False) -> list:
        """
        Collect image paths from a directory with extension filtering.
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process (e.g., ['.jpg', '.png']).
                       If None, uses default: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            recursive: Whether to search recursively in subdirectories
            
        Returns:
            Sorted list of image paths (unique)

        Raises:
            ValueError: If directory does not exist
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Default extensions if not provided TODO add more extensions
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Find all image files
        image_paths = []
        if recursive:
            # Search recursively
            for ext in extensions:
                image_paths.extend(dir_path.rglob(f'*{ext}'))
                image_paths.extend(dir_path.rglob(f'*{ext.upper()}'))
        else:
            # Search only in the directory
            for ext in extensions:
                image_paths.extend(dir_path.glob(f'*{ext}'))
                image_paths.extend(dir_path.glob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        if not image_paths:
            logger.warning(f"No images found in directory: {directory_path}")
        
        return image_paths
    
    def get_metrics_only(self, image_input: Union[str, Path, np.ndarray]) -> Dict:
        """
        Get only the metrics without quality analysis.
        
        Args:
            image_input: Path to image file or numpy array
            
        Returns:
            Dictionary with metrics: {'sharpness': float, 'exposure': float, 'brisque': Optional[float]}
        """
        image = self.load_image(image_input)
        
        metrics = {
            'sharpness': self._to_float(self.calculate_sharpness(image), DEFAULT_ROUND_DIGITS),
            'exposure': self._to_float(self.calculate_exposure(image), DEFAULT_ROUND_DIGITS)
        }
        
        # Add BRISQUE if IQA is available
        if self.use_iqa and self.iqa_metrics:
            try:
                metrics['brisque'] = self.iqa_metrics.calculate_brisque(image)
            except Exception as e:
                logger.warning(f"Could not calculate BRISQUE: {e}")
                metrics['brisque'] = None
        
        return metrics
    
    def _normalize_sharpness(self, sharpness: float, max_sharpness: float = 500.0) -> float:
        """
        Normalize sharpness value to 0-1 range.
        
        Args:
            sharpness: Raw sharpness value (Laplacian variance)
            max_sharpness: Maximum expected sharpness for normalization
            
        Returns:
            Normalized sharpness (0-1, where 1 = best)
        """
        # Normalize: clamp to [0, max_sharpness] and scale to [0, 1]
        normalized = min(1.0, max(0.0, sharpness / max_sharpness))
        return normalized
    
    def _normalize_exposure(self, exposure: float) -> float:
        """
        Normalize exposure to quality score (0-1).
        
        Exposure is already 0-1, but we want to penalize extremes.
        Best exposure is around 0.5 (middle gray).
        
        Args:
            exposure: Exposure value (0-1)
            
        Returns:
            Normalized exposure quality (0-1, where 1 = best)
        """
        # Ideal exposure is around 0.5 (middle gray)
        # Penalize extremes: too dark (close to 0) or too bright (close to 1)
        # Use a quadratic penalty centered at 0.5
        distance_from_ideal = abs(exposure - 0.5)
        quality = 1.0 - (distance_from_ideal * 2.0)  # Scale to [0, 1]
        return max(0.0, quality)
    
    def get_quality_score(self, image_input: Union[str, Path, np.ndarray],
                         weights: Optional[Dict[str, float]] = None,
                         max_sharpness: Optional[float] = None,
                         max_brisque: Optional[float] = None) -> Dict:
        """
        Calculate overall quality score (0-1) combining multiple metrics.
        
        Combines sharpness, exposure, and optionally BRISQUE into a single
        quality score where 1.0 = best quality, 0.0 = worst quality.
        
        Args:
            image_input: Path to image file or numpy array
            weights: Dictionary with weights for each metric.
                    Default: {'sharpness': 0.4, 'exposure': 0.3, 'brisque': 0.3}
                    If BRISQUE is not available, weights are renormalized.
            max_sharpness: Maximum expected sharpness for normalization. If None, uses recommended config.
            max_brisque: Maximum expected BRISQUE value for normalization. If None, uses recommended config.
            
        Returns:
            Dictionary with quality score and component scores:
            {
                'quality_score': float,  # Overall score (0-1)
                'sharpness_score': float,  # Normalized sharpness (0-1)
                'exposure_score': float,   # Normalized exposure quality (0-1)
                'brisque_score': Optional[float],  # Normalized BRISQUE (0-1) if available
                'metrics': {
                    'sharpness': float,    # Raw sharpness value
                    'exposure': float,     # Raw exposure value
                    'brisque': Optional[float]  # Raw BRISQUE value if available
                }
            }
        """
        # Load image
        image = self.load_image(image_input)
        max_sharpness = max_sharpness if max_sharpness is not None else self.max_sharpness_default
        max_brisque = max_brisque if max_brisque is not None else self.max_brisque_default
        
        # Calculate raw metrics
        sharpness = self.calculate_sharpness(image)
        exposure = self.calculate_exposure(image)
        
        # Normalize metrics to 0-1
        sharpness_score = self._normalize_sharpness(sharpness, max_sharpness)
        exposure_score = self._normalize_exposure(exposure)
        
        # Calculate BRISQUE if available
        brisque_score = None
        brisque_raw = None
        if self.use_iqa and self.iqa_metrics:
            try:
                brisque_raw = self.iqa_metrics.calculate_brisque(image)
                brisque_score = self.iqa_metrics.calculate_brisque_normalized(
                    image, max_brisque=max_brisque, brisque_raw=brisque_raw
                )
            except Exception as e:
                logger.warning(f"Could not calculate BRISQUE: {e}")
        
        # Set default weights
        if weights is None:
            if brisque_score is not None:
                weights = {'sharpness': 0.4, 'exposure': 0.3, 'brisque': 0.3}
            else:
                weights = {'sharpness': 0.6, 'exposure': 0.4}
        else:
            # Avoid mutating caller-provided weights
            weights = dict(weights)
        
        # If BRISQUE is not available, renormalize weights so they sum to 1.0
        if brisque_score is None and 'brisque' in weights:
            total_weight = weights['sharpness'] + weights['exposure']
            if total_weight > 0:
                weights['sharpness'] /= total_weight
                weights['exposure'] /= total_weight
            del weights['brisque']
        
        # Calculate weighted quality score
        quality_score = (
            weights.get('sharpness', 0.0) * sharpness_score +
            weights.get('exposure', 0.0) * exposure_score
        )
        
        if brisque_score is not None:
            quality_score += weights.get('brisque', 0.0) * brisque_score
        
        # Ensure score is in [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            'quality_score': self._to_float(quality_score, DEFAULT_ROUND_DIGITS),
            'sharpness_score': self._to_float(sharpness_score, DEFAULT_ROUND_DIGITS),
            'exposure_score': self._to_float(exposure_score, DEFAULT_ROUND_DIGITS),
            'brisque_score': self._to_float(brisque_score, DEFAULT_ROUND_DIGITS) if brisque_score is not None else None,
            'metrics': {
                'sharpness': self._to_float(sharpness, DEFAULT_ROUND_DIGITS),
                'exposure': self._to_float(exposure, DEFAULT_ROUND_DIGITS),
                'brisque': self._to_float(brisque_raw, DEFAULT_ROUND_DIGITS) if brisque_raw is not None else None
            }
        }
