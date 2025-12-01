import numpy as np
from scipy import ndimage
from skimage import filters
from typing import Tuple
import logging

logger = logging.getLogger("solar_masker.utils.features")

class FeatureUtils:
    """Static utilities for calculating image features."""
    
    @staticmethod
    def calculate_mean_intensity(image: np.ndarray, mask: np.ndarray) -> float:
        masked_pixels = image[mask > 0.5]
        if len(masked_pixels) == 0:
            logger.debug("calculate_mean_intensity: Empty mask provided.")
            return 0.0
        mean_val = np.mean(masked_pixels)
        return mean_val / 255.0 if image.dtype == np.uint8 else mean_val

    @staticmethod
    def calculate_intensity_variance(image: np.ndarray, mask: np.ndarray) -> float:
        masked_pixels = image[mask > 0.5]
        if len(masked_pixels) == 0:
            return 0.0
        variance = np.var(masked_pixels)
        return variance / (255.0 ** 2) if image.dtype == np.uint8 else variance

    @staticmethod
    def calculate_edge_density(image: np.ndarray, mask: np.ndarray) -> float:
        edges = filters.sobel(image)
        masked_edges = edges[mask > 0.5]
        if len(masked_edges) == 0:
            return 0.0
        return np.mean(masked_edges > 0.1)

    @staticmethod
    def count_mask_components(mask: np.ndarray) -> int:
        labeled, num_components = ndimage.label(mask > 0.5)
        return num_components

    @staticmethod
    def check_grid_pattern(mask: np.ndarray, expected_panel_size: Tuple[int, int] = (70, 40), tolerance: float = 0.3) -> bool:
        labeled, num_components = ndimage.label(mask > 0.5)
        if num_components < 2:
            return False
        
        panel_count = 0
        expected_w, expected_h = expected_panel_size
        
        for region_id in range(1, num_components + 1):
            coords = np.argwhere(labeled == region_id)
            if len(coords) < 10: continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            width, height = x_max - x_min, y_max - y_min
            
            w_match = (1 - tolerance) * expected_w <= width <= (1 + tolerance) * expected_w
            h_match = (1 - tolerance) * expected_h <= height <= (1 + tolerance) * expected_h
            
            if w_match and h_match:
                panel_count += 1
                
        if panel_count >= 6:
            logger.debug(f"Grid pattern detected: {panel_count} sub-panels found.")
            return True
        return False