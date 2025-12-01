from pathlib import Path
from PIL import Image
import numpy as np
import torch
import logging
from typing import Dict

logger = logging.getLogger("solar_masker.utils.io")

class PanelIO:
    @staticmethod
    def save_results(state: Dict, output_dir: str, only_boxes: bool = False,  only_masks: bool = False):
        if not state.get('final_boxes'):
            logger.info("No final results to save.")
            return

        try:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            base_name = Path(state['image_path']).stem
            
            # Save Binary Mask of Final Boxes
            final_boxes = state.get('final_boxes', [])
            image_shape = state.get('image_shape', (100, 100))
            
            if len(final_boxes) > 0 and only_boxes:
                box_mask = np.zeros(image_shape, dtype=np.uint8)
                for box in final_boxes:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    # Clip coordinates to image dimensions
                    x1 = max(0, min(x1, image_shape[1]))
                    y1 = max(0, min(y1, image_shape[0]))
                    x2 = max(0, min(x2, image_shape[1]))
                    y2 = max(0, min(y2, image_shape[0]))
                    
                    box_mask[y1:y2, x1:x2] = 255
                
                save_path = path / f"{base_name}_box_mask.png"
                Image.fromarray(box_mask).save(save_path)
                logger.debug(f"Saved box mask to {save_path}")
            
            # Save Merged Segmentation Mask
            final_masks = state.get('final_masks', [])
            if len(final_masks) > 0 and only_masks:
                merged = torch.stack(final_masks).any(dim=0).cpu().numpy()
                mask_img = (merged * 255).astype(np.uint8)
                
                save_path = path / f"{base_name}_seg_mask.png"
                Image.fromarray(mask_img).save(save_path)
                logger.debug(f"Saved segmentation mask to {save_path}")
                
            logger.info(f"Results saved to directory: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")