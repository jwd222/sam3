import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image
import logging
from typing import Dict

logger = logging.getLogger("solar_masker.utils.vis")

class PanelVisualizer:
    @staticmethod
    def visualize(image_path: str, state: Dict, save_path: str = None, show: bool = True):
        if not state.get('image_path'):
            logger.warning("Visualization skipped: No image path in state.")
            return

        try:
            image = Image.open(image_path).convert("RGB")
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: Intermediate
            ax = axes[0]
            ax.imshow(image, cmap="gray")
            ax.set_title("Intermediate (Stage 1 & 3)", fontweight='bold')
            
            if len(state.get('stage1_boxes', [])) > 0:
                for box in state['stage1_boxes']:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    
            if len(state.get('stage3_boxes', [])) > 0:
                 for box in state['stage3_boxes']:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)
            ax.axis('off')
            
            # Right: Final
            ax = axes[1]
            ax.imshow(image, cmap="gray")
            ax.set_title(f"Final Merged ({len(state.get('final_boxes', []))})", fontweight='bold')
            
            colors = ['lime', 'cyan', 'magenta', 'orange', 'yellow']
            final_boxes = state.get('final_boxes', [])
            final_masks = state.get('final_masks', [])
            
            for i in range(len(final_boxes)):
                color = colors[i % len(colors)]
                if i < len(final_masks):
                    mask_np = final_masks[i].cpu().numpy()
                    overlay = np.zeros((*mask_np.shape, 4))
                    overlay[mask_np > 0.5] = [*to_rgb(color), 0.35]
                    ax.imshow(overlay)
                
                x1, y1, x2, y2 = final_boxes[i].cpu().numpy()
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
            ax.axis('off')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
                
            if show:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            logger.error(f"Error during visualization: {e}")