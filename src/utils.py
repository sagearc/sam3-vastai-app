"""
Helper functions for SAM3 data construction and prompt handling.
"""

import os
from typing import List

import numpy as np
import torch
from PIL import Image
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    InferenceMetadata,
)
from sam3.train.data.sam3_image_dataset import Image as SAMImage


class SamPredictedSegmentationMask:
    def __init__(self, mask: torch.Tensor, score: torch.Tensor, bbox_mask: np.ndarray):
        self.mask = mask.squeeze().cpu().numpy().astype(np.uint8)
        self.score = score
        self.bbox_overlap = in_bbox_ratio(self.mask, bbox_mask)


class SamSegmentationResult:
    """
    Encapsulates SAM segmentation results with bbox overlap analysis.
    """

    def __init__(
        self,
        raw_results: dict,
        prompt: str,
        bbox: List[float],
        image_size: tuple[int, int],
    ):
        """
        Initialize segmentation result with overlap calculations.

        Args:
            raw_results: Dictionary containing 'masks' and 'scores' from SAM
            prompt: Text prompt used for segmentation
            bbox: Bounding box as [x1, y1, x2, y2]
            image_size: Image size as (width, height)
        """
        self.prompt = prompt
        self.bbox = bbox
        self.image_size = image_size
        self.segmentation_masks: list[SamPredictedSegmentationMask] = []

        bbox_mask = get_bbox_mask(bbox, image_size)
        for mask, score in zip(raw_results["masks"], raw_results["scores"]):
            seg = SamPredictedSegmentationMask(
                mask=mask,
                score=score,
                bbox_mask=bbox_mask,
            )
            self.segmentation_masks.append(seg)

    def __len__(self) -> int:
        """Return number of masks in result."""
        return len(self.segmentation_masks)

    def __str__(self) -> str:
        """String representation of the result."""
        return f"SamSegmentationResult(prompt='{self.prompt}', bbox={self.bbox}, n_masks={len(self.segmentation_masks)})"


def process_raw_segmentation_results(
    raw_results: list[dict],
    prompts: list[str],
    bboxes: list[list[float]],
    image_size: tuple,
) -> list[SamSegmentationResult]:
    all_segmentation_results = []
    for segmentation_res, prompt, bbox in zip(raw_results, prompts, bboxes):
        segmentation_results = SamSegmentationResult(
            raw_results=segmentation_res,
            prompt=prompt,
            bbox=bbox,
            image_size=image_size,
        )
        all_segmentation_results.append(segmentation_results)
    return all_segmentation_results


# --- Helper Functions for Data Construction ---
GLOBAL_COUNTER = 1


def create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def sample_points_from_mask(mask: torch.Tensor, num_points: int = 10) -> torch.Tensor:
    mask = torch.tensor(mask)
    # sample True points coordinates from mask tensor (nx2)
    points = torch.nonzero(mask.squeeze(), as_tuple=False)
    points = torch.flip(points, dims=[1])

    indices = torch.randperm(points.shape[0])[:num_points]
    points = points[indices]
    return points


def add_text_prompt(datapoint, text_query):
    global GLOBAL_COUNTER
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


def add_visual_prompt(
    datapoint, boxes: List[List[float]] | None, points: torch.Tensor | None, prompt: str
):
    global GLOBAL_COUNTER
    w, h = datapoint.images[0].size

    input_bbox = None
    input_bbox_label = None
    if boxes is not None:
        input_bbox = torch.tensor(boxes, dtype=torch.float).view(-1, 4)
        input_bbox_label = torch.ones(len(boxes), dtype=torch.bool)

    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=input_bbox,
            input_bbox_label=input_bbox_label,
            input_points=points,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


def calculate_mask_overlap(mask1, mask2):
    """
    Calculate the overlap percentage between two masks.

    Args:
        mask1: First mask tensor or numpy array
        mask2: Second mask tensor or numpy array

    Returns:
        Float representing the overlap percentage (0.0 to 1.0)
    """
    if torch.is_tensor(mask1):
        mask1_array = mask1.cpu().numpy()
    else:
        mask1_array = np.array(mask1)

    if torch.is_tensor(mask2):
        mask2_array = mask2.cpu().numpy()
    else:
        mask2_array = np.array(mask2)

    # Handle different tensor shapes
    if mask1_array.ndim == 3:
        mask1_array = mask1_array[0]
    elif mask1_array.ndim == 4:
        mask1_array = mask1_array[0, 0]

    if mask2_array.ndim == 3:
        mask2_array = mask2_array[0]
    elif mask2_array.ndim == 4:
        mask2_array = mask2_array[0, 0]

    # Convert to binary
    mask1_binary = mask1_array > 0.5
    mask2_binary = mask2_array > 0.5

    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary)
    mask1_area = np.sum(mask1_binary)

    if mask1_area == 0:
        return 0.0

    # Return overlap as percentage of first mask covered by intersection
    overlap_percentage = np.sum(intersection) / mask1_area
    return overlap_percentage


def is_mask_unique(new_mask, existing_masks, threshold=0.9):
    """
    Check if a new mask is sufficiently unique compared to existing masks.

    Args:
        new_mask: New mask tensor or numpy array
        existing_masks: List of existing mask tensors/arrays
        threshold: Minimum uniqueness threshold (0.5 = at most 50% overlap allowed)

    Returns:
        Boolean indicating if the mask is unique enough
    """
    if not existing_masks:
        return True

    for existing_mask in existing_masks:
        overlap = calculate_mask_overlap(new_mask, existing_mask)
        if overlap > threshold:
            return False
    return True


def get_bbox_mask(bbox: List[float], image_size: tuple) -> np.ndarray:
    """Create a binary mask for the given bounding box."""
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, bbox)
    mask[y1:y2, x1:x2] = 1
    return mask


def in_bbox_ratio(mask: np.ndarray, bbox_mask: np.ndarray) -> float:
    """Calculate the ratio of mask pixels inside the bounding box."""
    intersection = np.logical_and(mask, bbox_mask).sum()
    if mask.sum() == 0:
        return 0.0
    return intersection / mask.sum()


def save_mask_to_dir(mask_tensor, filename, object_label, score, output_dir):
    """Save a mask tensor as an image file in the specified output directory."""
    # Convert tensor to numpy array
    if torch.is_tensor(mask_tensor):
        mask_array = mask_tensor.cpu().numpy()
    else:
        mask_array = np.array(mask_tensor)

    # Handle different tensor shapes
    if mask_array.ndim == 3:
        mask_array = mask_array[0]  # Remove batch dimension if present
    elif mask_array.ndim == 4:
        mask_array = mask_array[0, 0]  # Remove batch and channel dimensions

    # Convert to 0-255 range
    mask_array = (mask_array * 255).astype(np.uint8)

    # Create PIL Image and save
    mask_image = Image.fromarray(mask_array, mode="L")

    # Create output directory if it doesn't exist
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Clean filename
    clean_label = "".join(
        c for c in object_label if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    clean_label = clean_label.replace(" ", "_")

    filepath = os.path.join(
        masks_dir, f"{filename}_{clean_label}_score_{score:.3f}.png"
    )
    mask_image.save(filepath)

    return filepath
