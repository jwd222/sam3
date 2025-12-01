"""
Unified SAM 3 Wrapper for Image Segmentation
Supports text, box, and point prompts in a single interface
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional, Dict
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox


class SAM3Wrapper:
    """
    Unified wrapper for SAM 3 image segmentation with all prompt types.
    
    Example Usage:
        # Initialize
        sam = SAM3Wrapper(checkpoint_path="checkpoints/sam3.pt")
        
        # Text-only
        results = sam.segment(image, text="person")
        
        # Box-only (positive)
        results = sam.segment(image, boxes=[[100, 100, 50, 50]])
        
        # Box with positive/negative
        results = sam.segment(
            image, 
            boxes=[[100, 100, 50, 50], [200, 200, 30, 30]],
            box_labels=[True, False]
        )
        
        # Points
        results = sam.segment(
            image,
            points=[[0.5, 0.5], [0.3, 0.3]],
            point_labels=[1, 0]
        )
        
        # Combined: text + boxes
        results = sam.segment(image, text="shoe", boxes=[[100, 100, 50, 50]])
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        bpe_path: str = "assets/bpe_simple_vocab_16e6.txt.gz",
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        resolution: int = 1008
    ):
        """
        Initialize SAM 3 model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            bpe_path: Path to BPE vocabulary file
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for predictions
            resolution: Input resolution for model
        """
        self.device = device
        
        # Enable optimizations
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        
        # Build model
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path
        )
        
        # Initialize processor
        self.processor = Sam3Processor(
            self.model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold
        )
        
        self.current_image = None
        self.inference_state = None
    
    def set_image(self, image: Union[str, Image.Image, np.ndarray]) -> None:
        """
        Set the image for inference.
        
        Args:
            image: PIL Image, numpy array, or path to image file
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        self.current_image = image
        self.inference_state = self.processor.set_image(image)
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.processor.set_confidence_threshold(threshold, self.inference_state)
    
    def segment(
        self,
        image: Optional[Union[str, Image.Image, np.ndarray]] = None,
        text: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[bool]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        box_format: str = "xywh",
        normalized: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Segment image with text, box, and/or point prompts.
        
        Args:
            image: Input image (if None, uses previously set image)
            text: Text prompt (e.g., "person", "shoe")
            boxes: List of boxes in format specified by box_format
            box_labels: Labels for boxes (True=positive, False=negative)
            points: List of [x, y] coordinates (normalized 0-1 or absolute)
            point_labels: Labels for points (1=positive, 0=negative)
            box_format: Format of input boxes - "xywh", "xyxy", or "cxcywh"
            normalized: Whether boxes/points are already normalized to [0,1]
        
        Returns:
            Dictionary with keys:
                - 'masks': Binary masks [N, H, W]
                - 'masks_logits': Raw mask logits [N, 1, H, W]
                - 'boxes': Bounding boxes in xyxy format [N, 4]
                - 'scores': Confidence scores [N]
        """
        # Set image if provided
        if image is not None:
            self.set_image(image)
        
        if self.inference_state is None:
            raise ValueError("No image set. Call set_image() or provide image parameter.")
        
        # Get image dimensions
        img_width, img_height = self.current_image.size
        
        # Reset prompts
        self.processor.reset_all_prompts(self.inference_state)
        
        # Add text prompt
        if text:
            self.inference_state = self.processor.set_text_prompt(
                state=self.inference_state,
                prompt=text
            )
        
        # Add box prompts
        if boxes:
            if box_labels is None:
                box_labels = [True] * len(boxes)
            
            for box, label in zip(boxes, box_labels):
                box_normalized = self._normalize_box(
                    box, img_width, img_height, box_format, normalized
                )
                self.inference_state = self.processor.add_geometric_prompt(
                    state=self.inference_state,
                    box=box_normalized,
                    label=label
                )
        
        # Note: Point prompts would need to be added via geometric_prompt
        # The current processor doesn't expose point prompts directly,
        # but they're supported in the underlying model
        if points:
            print("Warning: Point prompts not yet fully exposed in processor API")
        
        return {
            'masks': self.inference_state['masks'],
            'masks_logits': self.inference_state['masks_logits'],
            'boxes': self.inference_state['boxes'],
            'scores': self.inference_state['scores']
        }
    
    def _normalize_box(
        self,
        box: List[float],
        img_width: int,
        img_height: int,
        box_format: str,
        already_normalized: bool
    ) -> List[float]:
        """
        Normalize box to [center_x, center_y, width, height] in [0, 1] range.
        
        Args:
            box: Input box coordinates
            img_width: Image width
            img_height: Image height
            box_format: Format of input box - "xywh", "xyxy", or "cxcywh"
            already_normalized: Whether box is already normalized
        
        Returns:
            Normalized box in cxcywh format
        """
        box_tensor = torch.tensor(box).view(1, 4)
        
        # Convert to cxcywh if needed
        if box_format == "xywh":
            box_tensor = box_xywh_to_cxcywh(box_tensor)
        elif box_format == "xyxy":
            # xyxy to cxcywh
            x1, y1, x2, y2 = box_tensor[0]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            box_tensor = torch.tensor([[cx, cy, w, h]])
        elif box_format == "cxcywh":
            pass  # Already in correct format
        else:
            raise ValueError(f"Unknown box_format: {box_format}")
        
        # Normalize if needed
        if not already_normalized:
            box_tensor = normalize_bbox(box_tensor, img_width, img_height)
        
        return box_tensor.flatten().tolist()
    
    def batch_segment(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        text: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Segment multiple images with the same prompt.
        
        Args:
            images: List of images
            text: Text prompt to use for all images
            **kwargs: Additional arguments passed to segment()
        
        Returns:
            List of result dictionaries, one per image
        """
        results = []
        for image in images:
            result = self.segment(image=image, text=text, **kwargs)
            results.append(result)
        return results
    
    def get_largest_mask(self) -> Optional[torch.Tensor]:
        """Get the mask with largest area from last inference."""
        if self.inference_state is None or 'masks' not in self.inference_state:
            return None
        
        masks = self.inference_state['masks']
        if len(masks) == 0:
            return None
        
        areas = masks.sum(dim=(1, 2, 3))
        largest_idx = areas.argmax()
        return masks[largest_idx]
    
    def get_highest_confidence_mask(self) -> Optional[torch.Tensor]:
        """Get the mask with highest confidence score from last inference."""
        if self.inference_state is None or 'masks' not in self.inference_state:
            return None
        
        masks = self.inference_state['masks']
        scores = self.inference_state['scores']
        
        if len(masks) == 0:
            return None
        
        highest_idx = scores.argmax()
        return masks[highest_idx]