import torch
import logging
from typing import List

logger = logging.getLogger("solar_masker.utils.geometry")

class BoxUtils:
    """Static utilities for Bounding Box operations"""

    @staticmethod
    def calculate_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate Matrix IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)
        return iou

    @staticmethod
    def merge_connected_components(boxes: torch.Tensor, iou_threshold: float = 0.01) -> List[List[int]]:
        """Return list of indices for connected components based on IoU"""
        n = len(boxes)
        if n == 0: return []
        
        ious = BoxUtils.calculate_iou_pairwise(boxes, boxes)
        adjacency = (ious > iou_threshold).cpu().numpy()
        
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
                comp_indices = []
                dfs(i, comp_indices)
                components.append(comp_indices)
        
        logger.debug(f"Merged {n} boxes into {len(components)} connected components.")
        return components
