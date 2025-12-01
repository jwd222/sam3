import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict


class PanelIO:
    @staticmethod
    def save_results(state: Dict, output_dir: str, only_boxes: bool = False,  only_masks: bool = False):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        base_name = Path(state['image_path']).stem
        
        # Save Binary Mask of Final Boxes
        final_boxes = state.get('final_boxes', [])
        image_shape = state.get('image_shape', (100, 100)) # Default backup
        
        if len(final_boxes) > 0 and only_boxes:
            box_mask = np.zeros(image_shape, dtype=np.uint8)
            for box in final_boxes:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                box_mask[y1:y2, x1:x2] = 255
            
            Image.fromarray(box_mask).save(path / f"{base_name}_box_mask.png")
        
        # Save Merged Mask (actual segmentation)
        final_masks = state.get('final_masks', [])
        if len(final_masks) > 0 and only_masks:
            merged = torch.stack(final_masks).any(dim=0).cpu().numpy()
            mask_img = (merged * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(path / f"{base_name}_seg_mask.png")
            
        print(f"  Results saved to {output_dir}")
