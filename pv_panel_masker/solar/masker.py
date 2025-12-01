import torch
import numpy as np
from PIL import Image
from typing import Dict, Any

# Internal imports
from .utils.visualization import PanelVisualizer
from .utils.io import PanelIO
from .config import PanelMaskerConfig
from .utils.features import FeatureUtils
from .utils.geometry import BoxUtils

# ============================================================================
# MAIN CLASS: PANEL MASKER
# ============================================================================

class PanelMasker:
    """
    Main class for Solar Panel Segmentation.
    Stores intermediate results in `self.state` for introspection.
    """
    
    def __init__(self, sam_wrapper, config: PanelMaskerConfig = PanelMaskerConfig()):
        self.sam = sam_wrapper
        self.cfg = config
        self.state = {} # Stores intermediate results of the last run
        self.panel_visualizer = PanelVisualizer()
        self.panel_io = PanelIO()
        
    def reset_state(self):
        """Clear intermediate results."""
        self.state = {
            'image_path': None,
            'image_shape': None,
            'raw_boxes': [],
            'raw_masks': [],
            'stage1_boxes': [],
            'stage2_boxes': [],
            'stage3_boxes': [],
            'validation_reasons': [],
            'final_boxes': [],
            'final_masks': []
        }

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Full pipeline execution.
        Populates self.state with all intermediate and final results.
        """
        self.reset_state()
        print(f"\nProcessing: {image_path}")
        self.state['image_path'] = image_path
        
        # Load Image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        self.state['image_shape'] = image.size[::-1] # H, W
        
        # 0. SAM Inference
        self.sam.set_confidence_threshold(self.cfg.confidence_threshold)
        results = self.sam.segment(image=image, text=self.cfg.text_prompt)
        
        raw_boxes = results['boxes']
        raw_masks = results['masks'].squeeze(1)
        raw_scores = results['scores']
        
        self.state.update({
            'raw_boxes': raw_boxes,
            'raw_masks': raw_masks,
            'raw_scores': raw_scores
        })
        
        if len(raw_boxes) == 0:
            print("  No detections found.")
            return self.state

        # 1. Stage 1: Individual Panels
        s1_boxes, s1_masks, s1_scores = self._stage1_individual_panels(
            raw_boxes, raw_masks, raw_scores, image_array
        )
        self.state.update({
            'stage1_boxes': s1_boxes,
            'stage1_masks': s1_masks,
            'stage1_scores': s1_scores
        })

        # 2. Stage 2: Large Objects
        s2_boxes, s2_masks, s2_scores = self._stage2_large_objects(
            raw_boxes, raw_masks, raw_scores, image_array
        )
        self.state.update({
            'stage2_boxes': s2_boxes,
            'stage2_masks': s2_masks,
            'stage2_scores': s2_scores
        })

        # 3. Stage 3: Validation
        s3_boxes, s3_masks, s3_scores, reasons = self._stage3_validation(
            s2_boxes, s2_masks, s2_scores, image_array
        )
        self.state.update({
            'stage3_boxes': s3_boxes,
            'stage3_masks': s3_masks,
            'stage3_scores': s3_scores,
            'validation_reasons': reasons
        })

        # 4. Stage 4: Merging
        final_boxes, final_masks = self._stage4_merging(
            s1_boxes, s1_masks, s3_boxes, s3_masks
        )
        self.state.update({
            'final_boxes': final_boxes,
            'final_masks': final_masks
        })
        
        print(f"  Final: {len(final_boxes)} panels merged.")
        return self.state

    # ------------------------------------------------------------------------
    # Internal Stage Methods
    # ------------------------------------------------------------------------

    def _stage1_individual_panels(self, boxes, masks, scores, image_array):
        if len(boxes) == 0: return boxes, masks, scores
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        # Size Filter
        # size_mask = (
        #     ((widths >= self.cfg.indiv_min_width) & (widths <= self.cfg.indiv_max_width) &
        #     (heights >= self.cfg.indiv_min_height) & (heights <= self.cfg.indiv_max_height)) | 
        #     ((widths >= self.cfg.indiv_min_width) & (widths <= 2 * self.cfg.indiv_max_width) &
        #     (heights >= self.cfg.indiv_min_height) & (heights <= self.cfg.indiv_max_height)) 
        # ).to(torch.bool).to(boxes.device)
        
        size_mask = (
            ((widths <= self.cfg.indiv_max_width) & (heights <= self.cfg.indiv_max_height)) | 
            ((widths <= 2 * self.cfg.indiv_max_width) & (heights <= self.cfg.indiv_max_height)) 
        ).to(torch.bool).to(boxes.device)
        
        # Intensity Filter
        image_gray = image_array[:, :, 0] if image_array.ndim == 3 else image_array
        valid_indices = []
        
        for i in range(len(boxes)):
            if not size_mask[i]: continue
            
            mean_intensity = FeatureUtils.calculate_mean_intensity(image_gray, masks[i].cpu().numpy())
            if mean_intensity >= self.cfg.indiv_min_intensity:
                valid_indices.append(i)
                
        if not valid_indices:
            return torch.tensor([]).to(boxes.device), torch.tensor([]).to(boxes.device), torch.tensor([]).to(boxes.device)
            
        idxs = torch.tensor(valid_indices, device=boxes.device)
        return boxes[idxs], masks[idxs], scores[idxs]

    def _stage2_large_objects(self, boxes, masks, scores, image_array):
        if len(boxes) == 0: return boxes, masks, scores
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        size_mask = ((widths >= self.cfg.large_min_dim) | (heights >= self.cfg.large_min_dim)).to(torch.bool).to(boxes.device)
        
        image_gray = image_array[:, :, 0] if image_array.ndim == 3 else image_array
        valid_indices = []
        
        for i in range(len(boxes)):
            if not size_mask[i]: continue
            
            mean_intensity = FeatureUtils.calculate_mean_intensity(image_gray, masks[i].cpu().numpy())
            if mean_intensity >= self.cfg.large_min_intensity:
                valid_indices.append(i)

        if not valid_indices:
            return torch.tensor([]).to(boxes.device), torch.tensor([]).to(boxes.device), torch.tensor([]).to(boxes.device)

        idxs = torch.tensor(valid_indices, device=boxes.device)
        return boxes[idxs], masks[idxs], scores[idxs]

    def _stage3_validation(self, boxes, masks, scores, image_array):
        if len(boxes) == 0: return boxes, masks, scores, []
        
        image_gray = image_array[:, :, 0] if image_array.ndim == 3 else image_array
        valid_indices = []
        reasons_log = []
        
        for i in range(len(boxes)):
            mask_np = masks[i].cpu().numpy()
            mean_intensity = FeatureUtils.calculate_mean_intensity(image_gray, mask_np)
            
            # Heuristics
            is_valid = False
            reasons = []
            
            if mean_intensity >= self.cfg.val_intensity_high:
                is_valid = True
                reasons.append(f"High Intensity ({mean_intensity:.2f})")
            elif mean_intensity >= self.cfg.val_intensity_medium:
                checks = 0
                edge_den = FeatureUtils.calculate_edge_density(image_gray, mask_np)
                comps = FeatureUtils.count_mask_components(mask_np)
                
                if comps >= self.cfg.val_min_components: checks += 1
                if edge_den >= self.cfg.val_edge_density: checks += 1
                if FeatureUtils.check_grid_pattern(mask_np): checks += 1
                
                if checks >= 2:
                    is_valid = True
                    reasons.append("Medium Intensity + Structure")
            
            if is_valid:
                valid_indices.append(i)
                reasons_log.append(" | ".join(reasons))
                
        if not valid_indices:
            return torch.tensor([]).to(boxes.device), torch.tensor([]).to(boxes.device), torch.tensor([]).to(boxes.device), []
            
        idxs = torch.tensor(valid_indices, device=boxes.device)
        return boxes[idxs], masks[idxs], scores[idxs], reasons_log

    def _stage4_merging(self, box_s1, mask_s1, box_s3, mask_s3):
        """Merges individual panels (Stage 1) and validated arrays (Stage 3)"""
        if len(box_s1) == 0 and len(box_s3) == 0:
            return [], []
            
        # Concatenate available detections
        tensors_to_cat_box = [b for b in [box_s1, box_s3] if len(b) > 0]
        tensors_to_cat_mask = [m for m in [mask_s1, mask_s3] if len(m) > 0]
        
        all_boxes = torch.cat(tensors_to_cat_box, dim=0)
        all_masks = torch.cat(tensors_to_cat_mask, dim=0)
        
        # Connected Components
        component_indices = BoxUtils.merge_connected_components(all_boxes, self.cfg.merge_iou_threshold)
        
        final_boxes = []
        final_masks = []
        
        for indices in component_indices:
            # Indices is a list of python ints
            comp_boxes = all_boxes[indices]
            comp_masks = all_masks[indices]
            
            # Merge Geometries
            x1 = comp_boxes[:, 0].min()
            y1 = comp_boxes[:, 1].min()
            x2 = comp_boxes[:, 2].max()
            y2 = comp_boxes[:, 3].max()
            
            final_boxes.append(torch.tensor([x1, y1, x2, y2]))
            
            # Merge Masks
            if len(comp_masks) == 1:
                final_masks.append(comp_masks[0])
            else:
                final_masks.append(torch.stack(list(comp_masks)).any(dim=0))
                
        return final_boxes, final_masks

