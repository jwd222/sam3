from dataclasses import dataclass

@dataclass
class PanelMaskerConfig:
    """Configuration for Panel Masking Pipeline"""
    # SAM Settings
    text_prompt: str = "panel"
    confidence_threshold: float = 0.2
    
    # Stage 1: Individual Panel Filtering
    indiv_min_width: float = 60.0
    indiv_max_width: float = 80.0
    indiv_min_height: float = 35.0
    indiv_max_height: float = 50.0
    indiv_min_intensity: float = 0.35
    
    # Stage 2: Large Object Detection
    large_min_dim: float = 200.0
    large_min_intensity: float = 0.4
    
    # Stage 3: Advanced Validation
    val_intensity_high: float = 0.5
    val_intensity_medium: float = 0.35
    val_edge_density: float = 0.08
    val_min_components: int = 2
    
    # Stage 4: Merging
    merge_iou_threshold: float = 0.01
