"""
Solar Panel Segmentation Script
Segments individual solar panel modules and merges them into complete panels.
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
from scipy import ndimage
from skimage import filters

from sam3_wrapper import SAM3Wrapper  # Import the wrapper

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1, box2: Boxes in [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2_min - x1_max)
    inter_height = max(0, y2_min - y1_max)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def merge_boxes(boxes: torch.Tensor, iou_threshold: float = 0.01) -> List[List[int]]:
    """
    Merge overlapping/nearby boxes using connected components approach.
    
    Args:
        boxes: Tensor of boxes in [x1, y1, x2, y2] format [N, 4]
        iou_threshold: IoU threshold for merging (very low to catch nearby modules)
    
    Returns:
        List of merged box groups, each group is list of indices
    """
    n = len(boxes)
    if n == 0:
        return []
    
    # Build adjacency matrix
    adjacency = torch.zeros((n, n), dtype=torch.bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                adjacency[i, j] = True
                adjacency[j, i] = True
    
    # Find connected components using DFS
    visited = torch.zeros(n, dtype=torch.bool)
    components = []
    
    def dfs(node: int, component: List[int]):
        visited[node] = True
        component.append(node)
        neighbors = torch.where(adjacency[node])[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                dfs(neighbor.item(), component)
    
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    return components

def filter_boxes_by_size(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    max_width: float = 80.0,
    max_height: float = 50.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter out boxes that exceed maximum dimensions (removes shadows/artifacts).
    
    Args:
        boxes: Bounding boxes [N, 4] in xyxy format
        masks: Masks [N, H, W]
        scores: Confidence scores [N]
        max_width: Maximum allowed box width in pixels
        max_height: Maximum allowed box height in pixels
    
    Returns:
        Tuple of (filtered_boxes, filtered_masks, filtered_scores)
    """
    if len(boxes) == 0:
        return boxes, masks, scores
    
    # Calculate box dimensions
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    # Keep only boxes within size limits
    valid_mask = ((widths <= max_width) & (heights <= max_height)) | ((widths <= max_width * 2) & (heights <= max_height)) | ((widths <= max_width) & (heights <= max_height * 2))
    
    filtered_boxes = boxes[valid_mask]
    filtered_masks = masks[valid_mask]
    filtered_scores = scores[valid_mask]
    
    num_filtered = len(boxes) - len(filtered_boxes)
    if num_filtered > 0:
        print(f"  Filtered out {num_filtered} oversized boxes (likely shadows/artifacts)")
    
    return filtered_boxes, filtered_masks, filtered_scores

def get_merged_panel_boxes_and_masks(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    iou_threshold: float = 0.01
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Merge module boxes and masks into complete panel boxes and masks.
    
    Args:
        boxes: Module bounding boxes [N, 4] in xyxy format
        masks: Module masks [N, H, W]
        iou_threshold: IoU threshold for grouping modules into panels
    
    Returns:
        Tuple of (panel_boxes, panel_masks)
    """
    if len(boxes) == 0:
        return [], []
    
    # Get connected components (groups of modules belonging to same panel)
    components = merge_boxes(boxes, iou_threshold)
    
    panel_boxes = []
    panel_masks = []
    
    print(f"  Found {len(components)} panel groups from {len(boxes)} modules")
    
    for idx, component in enumerate(components):
        # Get boxes and masks for this component
        component_boxes = boxes[component]
        component_masks = masks[component]
        
        print(f"    Panel {idx+1}: {len(component)} modules")
        
        # Merge boxes by taking min/max coordinates
        x1_min = component_boxes[:, 0].min()
        y1_min = component_boxes[:, 1].min()
        x2_max = component_boxes[:, 2].max()
        y2_max = component_boxes[:, 3].max()
        
        merged_box = torch.tensor([x1_min, y1_min, x2_max, y2_max])
        
        # Merge masks by taking union (logical OR)
        # Use stack + any to properly combine all masks
        if len(component_masks) == 1:
            merged_mask = component_masks[0]
        else:
            merged_mask = torch.stack([component_masks[i] for i in range(len(component_masks))]).any(dim=0)
        
        panel_boxes.append(merged_box)
        panel_masks.append(merged_mask)
    
    return panel_boxes, panel_masks

def save_panel_data_original(
    image_name: str,
    panel_boxes: List[torch.Tensor],
    panel_masks: List[torch.Tensor],
    output_dir: str
):
    """
    Save panel segmentation data to disk.
    
    Args:
        image_name: Name of the original image
        panel_boxes: List of panel bounding boxes
        panel_masks: List of panel masks
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(image_name).stem
    
    # Save masks as PNG images
    for i, mask in enumerate(panel_masks):
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_np)
        mask_path = output_dir / f"{base_name}_panel_{i+1}_mask.png"
        mask_img.save(mask_path)
    
    # Save bounding boxes as text file
    boxes_path = output_dir / f"{base_name}_boxes.txt"
    with open(boxes_path, 'w') as f:
        f.write(f"# Image: {image_name}\n")
        f.write(f"# Format: panel_id x1 y1 x2 y2\n")
        for i, box in enumerate(panel_boxes):
            x1, y1, x2, y2 = box.cpu().numpy()
            f.write(f"{i+1} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
    
    print(f"Saved {len(panel_boxes)} panel masks and boxes to {output_dir}")


def visualize_results(
    image: Image.Image,
    module_boxes: torch.Tensor = None,
    panel_boxes: List[torch.Tensor] = None,
    panel_masks: List[torch.Tensor] = None,
    save_path: str = None,
    show_modules: bool = True,
    show_panels: bool = True,
    show_masks: bool = True,
    show_boxes: bool = True
):
    """
    Flexible visualizer with granular control over what to display.
    
    Args:
        image: PIL image
        module_boxes: Individual module boxes (optional)
        panel_boxes: Final merged panel boxes (optional)
        panel_masks: Final merged panel masks (optional)
        save_path: Optional save path
        show_modules: Whether to show left plot with individual modules
        show_panels: Whether to show right plot with final panels
        show_masks: Whether to show masks in panel plot
        show_boxes: Whether to show boxes in panel plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    
    # Determine what to show
    show_left = show_modules and module_boxes is not None and len(module_boxes) > 0
    show_right = show_panels and (
        (show_boxes and panel_boxes is not None and len(panel_boxes) > 0) or
        (show_masks and panel_masks is not None and len(panel_masks) > 0)
    )
    
    num_plots = show_left + show_right
    if num_plots == 0:
        print("Nothing to visualize.")
        return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))
    if num_plots == 1:
        axes = [axes]
    
    plot_index = 0
    
    # --------------------------------------------
    # LEFT PLOT — INDIVIDUAL MODULE BOXES
    # --------------------------------------------
    if show_left:
        ax = axes[plot_index]
        plot_index += 1
        
        ax.imshow(image, cmap="gray")
        n = len(module_boxes)
        ax.set_title(f'Individual Detections (n={n})', fontsize=14, fontweight='bold')
        
        for box in module_boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        ax.axis('off')
    
    # --------------------------------------------
    # RIGHT PLOT — FINAL PANELS (boxes and/or masks)
    # --------------------------------------------
    if show_right:
        ax = axes[plot_index]
        
        ax.imshow(image, cmap="gray")
        n_panels = max(
            len(panel_boxes) if panel_boxes else 0,
            len(panel_masks) if panel_masks else 0
        )
        ax.set_title(f'Final Panels (n={n_panels})', fontsize=14, fontweight='bold')
        
        colors = ['lime', 'cyan', 'yellow', 'magenta', 'orange', 'red', 'blue', 'white']
        
        for i in range(n_panels):
            color = colors[i % len(colors)]
            
            # Draw mask if requested and exists
            if show_masks and panel_masks and i < len(panel_masks):
                mask_np = panel_masks[i].cpu().numpy()
                overlay = np.zeros((*mask_np.shape, 4))
                overlay[mask_np > 0.5] = [*to_rgb(color), 0.35]
                ax.imshow(overlay)
            
            # Draw box if requested and exists
            if show_boxes and panel_boxes and i < len(panel_boxes):
                x1, y1, x2, y2 = panel_boxes[i].cpu().numpy()
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                ax.text(x1, y1 - 10, f'Panel {i+1}',
                       color=color, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    
    plt.show()


def save_panel_data(
    image_name: str,
    image_shape: Tuple[int, int],
    module_boxes: torch.Tensor,
    module_masks: torch.Tensor,
    panel_boxes: List[torch.Tensor],
    panel_masks: List[torch.Tensor],
    output_dir: str,
    save_boxes: bool = True,
    save_masks: bool = False,
    save_individual: bool = False,
    save_merged: bool = True
):
    """
    Save segmentation results with flexible options.
    
    Args:
        image_name: Name of the original image
        image_shape: (height, width) of the original image
        module_boxes: Individual module bounding boxes
        module_masks: Individual module masks
        panel_boxes: Final merged panel bounding boxes
        panel_masks: Final merged panel masks
        output_dir: Directory to save outputs
        save_boxes: Whether to save bounding boxes (recommended)
        save_masks: Whether to save masks
        save_individual: Whether to save individual detections
        save_merged: Whether to save final merged panels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(image_name).stem
    
    # ========================================================================
    # INDIVIDUAL DETECTIONS (from Stage 1 + Stage 2)
    # ========================================================================
    
    if save_individual:
        # Save individual module masks
        if save_masks and len(module_masks) > 0:
            # Combined mask
            individual_merged = module_masks.any(dim=0)
            mask_np = (individual_merged.cpu().numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_path = output_dir / f"{base_name}_mask.png"
            mask_img.save(mask_path)
            print(f"  Saved individual detections mask: {mask_path}")
        
        # # Save individual module boxes
        # if save_boxes and len(module_boxes) > 0:
        #     boxes_path = output_dir / f"{base_name}_individual_boxes.txt"
        #     with open(boxes_path, 'w') as f:
        #         f.write(f"# Image: {image_name}\n")
        #         f.write(f"# Shape: {image_shape[0]}x{image_shape[1]} (HxW)\n")
        #         f.write(f"# Format: detection_id x1 y1 x2 y2 width height\n")
        #         for i, box in enumerate(module_boxes):
        #             x1, y1, x2, y2 = box.cpu().numpy()
        #             w, h = x2 - x1, y2 - y1
        #             f.write(f"{i+1} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {w:.2f} {h:.2f}\n")
        #     print(f"  Saved individual detections boxes: {boxes_path}")
    
    # ========================================================================
    # FINAL MERGED PANELS (from Stage 4)
    # ========================================================================
    
    if save_merged:
        # Save merged panel masks
        if save_masks and len(panel_masks) > 0:
            # Combined mask
            merged_mask = torch.stack(panel_masks).any(dim=0)
            mask_np = (merged_mask.cpu().numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_path = output_dir / f"{base_name}_merged_mask.png"
            mask_img.save(mask_path)
            print(f"  Saved final panels mask: {mask_path}")
            
            # Individual panel masks (optional, can be disabled)
            # for i, mask in enumerate(panel_masks):
            #     mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            #     mask_img = Image.fromarray(mask_np)
            #     mask_path = output_dir / f"{base_name}_panel_{i+1}_mask.png"
            #     mask_img.save(mask_path)
        
        # Save merged panel boxes as binary mask images (RECOMMENDED PRIMARY OUTPUT)
        if save_boxes and len(panel_boxes) > 0:
            # Create binary mask from boxes
            box_mask = np.zeros(image_shape, dtype=np.uint8)
            
            for box in panel_boxes:
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                box_mask[y1:y2, x1:x2] = 255
            
            # Save as binary image
            mask_img = Image.fromarray(box_mask)
            mask_path = output_dir / f"{base_name}_box_mask.png"
            mask_img.save(mask_path)
            print(f"  Saved final panels boxes as mask: {mask_path}")            
        
def process_single_image_v1(
    sam_wrapper,
    image_path: str,
    text_prompt: str = "panel",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.01,
    max_module_width: float = 80.0,  # ADD THIS
    max_module_height: float = 50.0,  # ADD THIS
    visualize: bool = True,
    save_outputs: bool = True,
    output_dir: str = "panel_outputs"
) -> Dict:
    """
    Process a single image to segment solar panels.
    
    Args:
        sam_wrapper: Initialized SAM3Wrapper instance
        image_path: Path to image file
        text_prompt: Text prompt for segmentation
        confidence_threshold: Confidence threshold for SAM3
        iou_threshold: IoU threshold for merging modules
        max_module_width: Maximum width of a valid module in pixels
        max_module_height: Maximum height of a valid module in pixels
        visualize: Whether to show visualization
        save_outputs: Whether to save masks and boxes
        output_dir: Directory for saving outputs
    
    Returns:
        Dictionary with processing results
    """
    print(f"\nProcessing: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Set confidence threshold
    sam_wrapper.set_confidence_threshold(confidence_threshold)
    
    # Segment modules
    results = sam_wrapper.segment(image=image, text=text_prompt)
    
    module_boxes = results['boxes']
    module_masks = results['masks'].squeeze(1)  # Remove channel dim
    module_scores = results['scores']
    
    print(f"Found {len(module_boxes)} modules")
    
    # Filter out oversized boxes (shadows, artifacts)
    module_boxes, module_masks, module_scores = filter_boxes_by_size(
        module_boxes, module_masks, module_scores,
        max_width=max_module_width,
        max_height=max_module_height
    )
    
    print(f"After filtering: {len(module_boxes)} valid modules")
    
    # Merge modules into panels
    panel_boxes, panel_masks = get_merged_panel_boxes_and_masks(
        module_boxes, module_masks, iou_threshold
    )
    
    print(f"Merged into {len(panel_boxes)} panels")
    
    # Visualize
    if visualize:
        vis_path = None
        if save_outputs:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            base_name = Path(image_path).stem
            vis_path = str(output_path / f"{base_name}_visualization.png")
        
        visualize_results(image, module_boxes, panel_boxes, panel_masks)
    
    # Save outputs
    if save_outputs:
        save_panel_data(
            Path(image_path).name,
            image.size[::-1],  # (height, width)
            module_boxes,
            module_masks,
            panel_boxes,
            panel_masks,
            output_dir
        )
    
    return {
        'image_path': image_path,
        'num_modules': len(module_boxes),
        'num_panels': len(panel_boxes),
        'module_boxes': module_boxes,
        'panel_boxes': panel_boxes,
        'panel_masks': panel_masks
    }

# ============================================================================
# STAGE 1: INDIVIDUAL PANEL DETECTION
# ============================================================================

def filter_individual_panels(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    image_array: np.ndarray,
    min_width: float = 60.0,
    max_width: float = 80.0,
    min_height: float = 35.0,
    max_height: float = 50.0,
    min_intensity_percentile: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, masks, scores

    device = boxes.device

    # Calculate dimensions (torch tensors)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Ensure boolean tensor on same device
    size_mask = (
        (widths >= min_width) & (widths <= max_width) &
        (heights >= min_height) & (heights <= max_height)
    ).to(torch.bool).to(device)

    # Intensity validation - boolean tensor on same device
    intensity_mask = torch.zeros(len(boxes), dtype=torch.bool, device=device)

    # Get single channel if grayscale with 3 bands (keep as numpy)
    if image_array.ndim == 3:
        image_gray = image_array[:, :, 0]
    else:
        image_gray = image_array

    for i in range(len(boxes)):
        # if size fails, skip (keeps intensity_mask[i] False)
        if not bool(size_mask[i].item()):
            continue

        # Calculate mean intensity in masked region (mask_np must be numpy)
        mask_np = masks[i].cpu().numpy()  # ensure numpy
        mean_intensity = calculate_mean_intensity(image_gray, mask_np)

        # Convert result to plain Python bool BEFORE assigning to torch tensor
        is_bright = float(mean_intensity) >= float(min_intensity_percentile)
        intensity_mask[i] = bool(is_bright)

    # Now combine masks; ensure both are torch.bool and on same device
    valid_mask = torch.logical_and(size_mask.to(device), intensity_mask.to(device))

    filtered_boxes = boxes[valid_mask]
    filtered_masks = masks[valid_mask]
    filtered_scores = scores[valid_mask]

    print(f"  Stage 1: {valid_mask.sum().item()}/{len(boxes)} individual panels (size + intensity)")

    return filtered_boxes, filtered_masks, filtered_scores


# ============================================================================
# STAGE 2: LARGE OBJECT DETECTION
# ============================================================================

def filter_large_objects(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    image_array: np.ndarray,
    min_dimension: float = 200.0,
    min_mean_intensity: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if len(boxes) == 0:
        return boxes, masks, scores

    # Torch device
    device = boxes.device

    # Calculate width / height
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Size mask (torch.bool)
    size_mask = ((widths >= min_dimension) | (heights >= min_dimension)).to(torch.bool).to(device)

    # Convert image to grayscale numpy
    if image_array.ndim == 3:
        image_gray = image_array[:, :, 0]
    else:
        image_gray = image_array

    intensity_mask = torch.zeros(len(boxes), dtype=torch.bool, device=device)

    for i in range(len(boxes)):
        if not bool(size_mask[i].item()):
            continue

        mask_np = masks[i].cpu().numpy()
        mean_intensity = calculate_mean_intensity(image_gray, mask_np)

        # Convert numpy.bool_ comparison to Python bool
        intensity_mask[i] = bool(mean_intensity >= min_mean_intensity)

    # USE logical_and (torch)
    valid_mask = torch.logical_and(size_mask, intensity_mask)

    filtered_boxes = boxes[valid_mask]
    filtered_masks = masks[valid_mask]
    filtered_scores = scores[valid_mask]

    print(f"  Stage 2: {valid_mask.sum()}/{size_mask.sum()} large objects passed intensity filter")

    return filtered_boxes, filtered_masks, filtered_scores


# ============================================================================
# STAGE 3: ADVANCED SHADOW FILTERING
# ============================================================================

def validate_large_objects(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    image_array: np.ndarray,
    intensity_threshold_high: float = 0.6,
    intensity_threshold_medium: float = 0.4,
    edge_density_threshold: float = 0.1,
    min_components: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:

    if len(boxes) == 0:
        return boxes, masks, scores, []

    device = boxes.device

    # Convert to grayscale numpy
    if image_array.ndim == 3:
        image_gray = image_array[:, :, 0]
    else:
        image_gray = image_array

    # Torch.bool tensor
    valid_mask = torch.zeros(len(boxes), dtype=torch.bool, device=device)

    validation_reasons = []

    for i in range(len(boxes)):
        mask_np = masks[i].cpu().numpy()

        mean_intensity = calculate_mean_intensity(image_gray, mask_np)
        intensity_variance = calculate_intensity_variance(image_gray, mask_np)
        edge_density = calculate_edge_density(image_gray, mask_np)
        num_components = count_mask_components(mask_np)
        has_grid_pattern = check_grid_pattern(mask_np, expected_panel_size=(70, 40))

        reasons = []
        is_valid = False

        # -------- HIGH INTENSITY --------
        if mean_intensity >= intensity_threshold_high:
            is_valid = True
            reasons.append(f"high_intensity({mean_intensity:.2f})")

        # -------- MEDIUM INTENSITY --------
        elif mean_intensity >= intensity_threshold_medium:

            checks = 0

            if num_components >= min_components:
                checks += 1
                reasons.append(f"multi_component({num_components})")

            if edge_density >= edge_density_threshold:
                checks += 1
                reasons.append(f"high_edges({edge_density:.3f})")

            if has_grid_pattern:
                checks += 1
                reasons.append("grid_pattern")

            if intensity_variance > 0.05:
                checks += 1
                reasons.append(f"high_variance({intensity_variance:.3f})")

            if checks >= 2:
                is_valid = True
                reasons.append(f"medium_intensity({mean_intensity:.2f})")
            else:
                reasons.append(f"REJECTED_insufficient_evidence({checks}/2)")

        else:
            reasons.append(f"REJECTED_low_intensity({mean_intensity:.2f})")

        # Convert to Python bool before assigning
        valid_mask[i] = bool(is_valid)
        validation_reasons.append(" | ".join(reasons))

    validated_boxes = boxes[valid_mask]
    validated_masks = masks[valid_mask]
    validated_scores = scores[valid_mask]
    validated_reasons = [r for idx, r in enumerate(validation_reasons) if valid_mask[idx]]

    print(f"  Stage 3: {valid_mask.sum()}/{len(boxes)} large objects validated")
    for i, reason in enumerate(validated_reasons):
        print(f"    Object {i+1}: {reason}")

    return validated_boxes, validated_masks, validated_scores, validated_reasons


# ============================================================================
# FEATURE CALCULATION FUNCTIONS
# ============================================================================

def calculate_mean_intensity(image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate mean intensity in masked region, normalized to [0, 1]."""
    masked_pixels = image[mask > 0.5]
    if len(masked_pixels) == 0:
        return 0.0
    mean_val = np.mean(masked_pixels)
    # Normalize to [0, 1]
    return mean_val / 255.0 if image.dtype == np.uint8 else mean_val


def calculate_intensity_variance(image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate intensity variance in masked region."""
    masked_pixels = image[mask > 0.5]
    if len(masked_pixels) == 0:
        return 0.0
    variance = np.var(masked_pixels)
    # Normalize
    return variance / (255.0 ** 2) if image.dtype == np.uint8 else variance


def calculate_edge_density(image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate edge density using Sobel filter."""
    # Apply Sobel edge detection
    edges = filters.sobel(image)
    
    # Calculate edge density in masked region
    masked_edges = edges[mask > 0.5]
    if len(masked_edges) == 0:
        return 0.0
    
    # Ratio of strong edges
    edge_density = np.mean(masked_edges > 0.1)
    return edge_density


def count_mask_components(mask: np.ndarray) -> int:
    """Count connected components in mask."""
    labeled, num_components = ndimage.label(mask > 0.5)
    return num_components


def check_grid_pattern(
    mask: np.ndarray,
    expected_panel_size: Tuple[int, int] = (70, 40),
    tolerance: float = 0.3
) -> bool:
    """
    Check if mask contains a grid pattern of panel-sized regions.
    
    Args:
        mask: Binary mask
        expected_panel_size: Expected (width, height) of individual panels
        tolerance: Size tolerance (0.3 = ±30%)
    
    Returns:
        True if grid pattern detected
    """
    # Label connected components
    labeled, num_components = ndimage.label(mask > 0.5)
    
    if num_components < 2:
        return False
    
    # Check component sizes
    panel_count = 0
    expected_w, expected_h = expected_panel_size
    
    for region_id in range(1, num_components + 1):
        region_mask = (labeled == region_id)
        coords = np.argwhere(region_mask)
        
        if len(coords) < 10:  # Too small
            continue
        
        # Get bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Check if matches expected panel size
        w_match = (1 - tolerance) * expected_w <= width <= (1 + tolerance) * expected_w
        h_match = (1 - tolerance) * expected_h <= height <= (1 + tolerance) * expected_h
        
        if w_match and h_match:
            panel_count += 1
    
    # Should have at least 6 panels for an array
    return panel_count >= 6


# ============================================================================
# STAGE 4: MERGING
# ============================================================================

def merge_all_detections(
    individual_boxes: torch.Tensor,
    individual_masks: torch.Tensor,
    array_boxes: torch.Tensor,
    array_masks: torch.Tensor,
    iou_threshold: float = 0.01
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Stage 4: Merge individual panels and validated arrays.
    
    Args:
        individual_boxes: Individual panel boxes from Stage 1
        individual_masks: Individual panel masks from Stage 1
        array_boxes: Validated array boxes from Stage 3
        array_masks: Validated array masks from Stage 3
        iou_threshold: IoU threshold for merging
    
    Returns:
        Tuple of (final_boxes, final_masks)
    """
    # Combine all boxes and masks
    if len(individual_boxes) > 0 and len(array_boxes) > 0:
        all_boxes = torch.cat([individual_boxes, array_boxes], dim=0)
        all_masks = torch.cat([individual_masks, array_masks], dim=0)
    elif len(individual_boxes) > 0:
        all_boxes = individual_boxes
        all_masks = individual_masks
    elif len(array_boxes) > 0:
        all_boxes = array_boxes
        all_masks = array_masks
    else:
        return [], []
    
    print(f"  Stage 4: Merging {len(individual_boxes)} panels + {len(array_boxes)} arrays")
    
    # Get connected components
    components = merge_boxes(all_boxes, iou_threshold)
    
    final_boxes = []
    final_masks = []
    
    for idx, component in enumerate(components):
        component_boxes = all_boxes[component]
        component_masks = all_masks[component]
        
        # Merge boxes
        x1_min = component_boxes[:, 0].min()
        y1_min = component_boxes[:, 1].min()
        x2_max = component_boxes[:, 2].max()
        y2_max = component_boxes[:, 3].max()
        
        merged_box = torch.tensor([x1_min, y1_min, x2_max, y2_max])
        
        # Merge masks
        if len(component_masks) == 1:
            merged_mask = component_masks[0]
        else:
            merged_mask = torch.stack(list(component_masks)).any(dim=0)
        
        final_boxes.append(merged_box)
        final_masks.append(merged_mask)
        
        width = (x2_max - x1_min).item()
        height = (y2_max - y1_min).item()
        print(f"    Final mask {idx+1}: {len(component)} detections → {width:.1f}×{height:.1f}px")
    
    return final_boxes, final_masks


def merge_boxes(boxes: torch.Tensor, iou_threshold: float = 0.01) -> List[torch.Tensor]:
    """Group boxes into connected components based on IoU overlap."""
    n = len(boxes)
    if n == 0:
        return []
    
    # Compute pairwise IoU
    ious = box_iou(boxes, boxes)
    
    # Create adjacency matrix
    adjacency = (ious > iou_threshold).cpu().numpy()
    
    # Find connected components using DFS
    visited = [False] * n
    components = []
    
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in range(n):
            if adjacency[node, neighbor] and not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(torch.tensor(component))
    
    return components


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between two sets of boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_single_image(
    sam_wrapper,
    image_path: str,
    text_prompt: str = "panel",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.01,
    # Stage 1: Individual panels
    individual_panel_min_width: float = 60.0,
    individual_panel_max_width: float = 80.0,
    individual_panel_min_height: float = 35.0,
    individual_panel_max_height: float = 50.0,
    individual_panel_min_intensity: float = 0.5,
    # Stage 2: Large objects
    large_object_min_dimension: float = 200.0,
    large_object_min_intensity: float = 0.4,
    # Stage 3: Validation
    validation_intensity_high: float = 0.5,
    validation_intensity_medium: float = 0.35,
    validation_edge_density: float = 0.08,
    validation_min_components: int = 2,
    # Output options
    visualize: bool = True,
    save_outputs: bool = True,
    output_dir: str = "panel_outputs"
) -> Dict:
    """
    Process a single image with multi-stage panel detection and shadow filtering.
    
    Pipeline:
        Stage 1: Detect individual panels (strict size + intensity filter)
        Stage 2: Detect large objects (arrays/blocks) with basic intensity filter
        Stage 3: Validate large objects (advanced shadow filtering)
        Stage 4: Merge all validated detections
    
    Args:
        sam_wrapper: Initialized SAM3Wrapper instance
        image_path: Path to image file
        text_prompt: Text prompt for segmentation
        confidence_threshold: Confidence threshold for SAM3
        iou_threshold: IoU threshold for merging
        individual_panel_*: Parameters for Stage 1
        large_object_*: Parameters for Stage 2
        validation_*: Parameters for Stage 3
        visualize: Whether to show visualization
        save_outputs: Whether to save masks and boxes
        output_dir: Directory for saving outputs
    
    Returns:
        Dictionary with processing results
    """
    print(f"\n{'='*80}")
    print(f"Processing: {image_path}")
    print(f"{'='*80}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # Set confidence threshold
    sam_wrapper.set_confidence_threshold(confidence_threshold)
    
    # Initial segmentation
    print("\n[Initial Detection]")
    results = sam_wrapper.segment(image=image, text=text_prompt)
    
    all_boxes = results['boxes']
    all_masks = results['masks'].squeeze(1)
    all_scores = results['scores']
    
    print(f"  SAM3 found {len(all_boxes)} detections")
    
    if len(all_boxes) == 0:
        print("  No detections found!")
        return {
            'image_path': image_path,
            'num_detections': 0,
            'num_panels': 0,
            'panel_boxes': [],
            'panel_masks': []
        }
    
    # ========================================================================
    # STAGE 1: Individual Panel Detection
    # ========================================================================
    print("\n[Stage 1: Individual Panel Detection]")
    individual_boxes, individual_masks, individual_scores = filter_individual_panels(
        all_boxes, all_masks, all_scores, image_array,
        min_width=individual_panel_min_width,
        max_width=individual_panel_max_width,
        min_height=individual_panel_min_height,
        max_height=individual_panel_max_height,
        min_intensity_percentile=individual_panel_min_intensity
    )
    
    # ========================================================================
    # STAGE 2: Large Object Detection
    # ========================================================================
    print("\n[Stage 2: Large Object Detection]")
    large_boxes, large_masks, large_scores = filter_large_objects(
        all_boxes, all_masks, all_scores, image_array,
        min_dimension=large_object_min_dimension,
        min_mean_intensity=large_object_min_intensity
    )
    
    # ========================================================================
    # STAGE 3: Advanced Shadow Filtering
    # ========================================================================
    print("\n[Stage 3: Advanced Shadow Filtering]")
    validated_boxes, validated_masks, validated_scores, validation_reasons = validate_large_objects(
        large_boxes, 
        large_masks, 
        large_scores, 
        image_array,
        intensity_threshold_high=validation_intensity_high,
        intensity_threshold_medium=validation_intensity_medium,
        edge_density_threshold=validation_edge_density,
        min_components=validation_min_components
    )
    
    # ========================================================================
    # STAGE 4: Merging
    # ========================================================================
    print("\n[Stage 4: Merging Detections]")
    panel_boxes, panel_masks = merge_all_detections(
        individual_boxes, individual_masks,
        validated_boxes, validated_masks,
        iou_threshold=iou_threshold
    )
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {len(panel_boxes)} panel masks")
    print(f"{'='*80}")
    
    # Visualize
    if visualize:
        vis_path = None
        if save_outputs:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            base_name = Path(image_path).stem
            vis_path = str(output_path / f"{base_name}_visualization.png")
        
        # Flexible visualization
        visualize_results(
            image=image,
            module_boxes=individual_boxes,  # Optional
            panel_boxes=panel_boxes,         # Optional
            panel_masks=panel_masks,         # Optional
            save_path=None,
            show_modules=True,    # Show left plot (individual detections)
            show_panels=True,     # Show right plot (final panels)
            show_masks=True,      # Show masks in panel plot
            show_boxes=True       # Show boxes in panel plot
        )    
        
    # Save outputs
    if save_outputs:
        # Combine individual panels and validated arrays for "module" view
        all_detected_boxes = torch.cat([individual_boxes, validated_boxes], dim=0) if len(individual_boxes) > 0 and len(validated_boxes) > 0 else (individual_boxes if len(individual_boxes) > 0 else validated_boxes)
        all_detected_masks = torch.cat([individual_masks, validated_masks], dim=0) if len(individual_masks) > 0 and len(validated_masks) > 0 else (individual_masks if len(individual_masks) > 0 else validated_masks)
        
        save_panel_data(
            Path(image_path).name,
            image.size[::-1],
            all_detected_boxes,
            all_detected_masks,
            panel_boxes,
            panel_masks,
            output_dir,
            save_boxes=True,        # Save bounding boxes (RECOMMENDED)
            save_masks=True,       # Save masks (has holes, optional)
            save_individual=False,  # Save Stage 1+2 detections
            save_merged=True        # Save final merged panels (Stage 4)
        )
    
    return {
        'image_path': image_path,
        'num_detections': len(all_boxes),
        'num_individual_panels': len(individual_boxes),
        'num_validated_arrays': len(validated_boxes),
        'num_final_masks': len(panel_boxes),
        'individual_boxes': individual_boxes,
        'validated_boxes': validated_boxes,
        'panel_boxes': panel_boxes,
        'panel_masks': panel_masks,
        'validation_reasons': validation_reasons
    }
    

def process_image_folder(
    image_folder: str,
    checkpoint_path: str,
    bpe_path: str = "assets/bpe_simple_vocab_16e6.txt.gz",
    text_prompt: str = "panel",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.1,
    max_module_width: float = 80.0,  
    max_module_height: float = 50.0,  
    output_dir: str = "panel_outputs",
    visualize_all: bool = False,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
):
    """
    Process all images in a folder to segment solar panels.
    
    Args:
        image_folder: Path to folder containing images
        checkpoint_path: Path to SAM3 checkpoint
        bpe_path: Path to BPE vocabulary
        text_prompt: Text prompt for segmentation
        confidence_threshold: Confidence threshold for SAM3
        iou_threshold: IoU threshold for merging modules
        output_dir: Directory for saving outputs
        visualize_all: Whether to visualize all images (can be slow)
        image_extensions: List of valid image extensions
    """
    
    
    # Initialize SAM3
    print("Initializing SAM3 model...")
    sam = SAM3Wrapper(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        confidence_threshold=confidence_threshold
    )
    print("Model loaded successfully!")
    
    # Get all image files
    image_folder = Path(image_folder)
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f"*{ext}")))
        image_files.extend(list(image_folder.glob(f"*{ext.upper()}")))
    
    image_files = sorted(image_files)
    print(f"\nFound {len(image_files)} images in {image_folder}")
    
    if len(image_files) == 0:
        print("No images found! Check the folder path and image extensions.")
        return
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Image {i}/{len(image_files)}")
        
        try:
            result = process_single_image(
                sam_wrapper=sam,
                image_path=str(image_path),
                text_prompt=text_prompt,
                confidence_threshold=confidence_threshold,
                max_module_width=max_module_width,
                max_module_height=max_module_height,
                iou_threshold=iou_threshold,
                visualize=visualize_all,
                save_outputs=True,
                output_dir=output_dir
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {len(all_results)}")
    print(f"Total modules detected: {sum(r['num_modules'] for r in all_results)}")
    print(f"Total panels detected: {sum(r['num_panels'] for r in all_results)}")
    print(f"Average panels per image: {np.mean([r['num_panels'] for r in all_results]):.2f}")
    print(f"\nOutputs saved to: {output_dir}")


# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    IMAGE_FOLDER = "/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720/"
    CHECKPOINT_PATH = "checkpoints/sam3.pt"
    BPE_PATH = "assets/bpe_simple_vocab_16e6.txt.gz"
    OUTPUT_DIR = "/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720_sam3_masks_2/"
    
    # Segmentation parameters
    TEXT_PROMPT = "panel"
    CONFIDENCE_THRESHOLD = 0.2  # Adjust between 0.2-0.3 based on your images
    IOU_THRESHOLD = 0.01  # Very low to catch nearby modules
    
    MAX_MODULE_WIDTH = 80.0  # ADD THIS
    MAX_MODULE_HEIGHT = 50.0  # ADD THIS

    # Processing options
    VISUALIZE_ALL = False  # Set True to visualize every image (slower)
    
    # # Run processing
    # process_image_folder(
    #     image_folder=IMAGE_FOLDER,
    #     checkpoint_path=CHECKPOINT_PATH,
    #     bpe_path=BPE_PATH,
    #     text_prompt=TEXT_PROMPT,
    #     confidence_threshold=CONFIDENCE_THRESHOLD,
    #     iou_threshold=IOU_THRESHOLD,
    #     max_module_width=MAX_MODULE_WIDTH, 
    #     max_module_height=MAX_MODULE_HEIGHT, 
    #     output_dir=OUTPUT_DIR,
    #     visualize_all=VISUALIZE_ALL
    # )
    
    # ========================================================================
    # ALTERNATIVE: Process a single image for testing
    # ========================================================================
    from sam3_wrapper import SAM3Wrapper
    
    sam = SAM3Wrapper(
        checkpoint_path=CHECKPOINT_PATH,
        bpe_path=BPE_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # result = process_single_image_v1(
    #     sam_wrapper=sam,
    #     image_path="/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720/IRX_4933.JPG", # "D:\S\ANTS\Data\spectra_thermal\132FTASK_1761021720\IRX_4552.JPG"
    #     text_prompt=TEXT_PROMPT,
    #     confidence_threshold=CONFIDENCE_THRESHOLD,
    #     iou_threshold=IOU_THRESHOLD,
    #     max_module_width=MAX_MODULE_WIDTH,  
    #     max_module_height=MAX_MODULE_HEIGHT,  
    #     visualize=True,
    #     save_outputs=True,
    #     output_dir=OUTPUT_DIR
    # )
    
    result = process_single_image(
        sam_wrapper=sam,
        image_path="/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720/IRX_4570.JPG", # "D:\S\ANTS\Data\spectra_thermal\132FTASK_1761021720\IRX_4552.JPG"
        confidence_threshold=CONFIDENCE_THRESHOLD,
        # Stage 1: Individual panels (strict)
        individual_panel_max_width=80.0,
        individual_panel_max_height=50.0,
        individual_panel_min_intensity=0.5,  # Bright
        # Stage 2: Large objects (permissive)
        large_object_min_dimension=200.0,
        large_object_min_intensity=0.4,  # Moderate
        # Stage 3: Validation (smart)
        validation_intensity_high=0.5,   # Auto-pass if very bright
        validation_edge_density=0.08,     # Must have structure
        visualize=False,
        save_outputs=True,
        output_dir=OUTPUT_DIR
    )