"""
Pydantic models for segmentation metadata and results.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from PIL import Image
import numpy as np


class BoundingBox(BaseModel):
    """Bounding box representation."""
    coordinates: List[float] = Field(..., description="Bounding box")
    xyxy: bool = Field(..., description="Whether the box is in [x1, y1, x2, y2] format")
    
    def flip_coordinates(self) -> None:
        bbox = self.coordinates
        self.coordinates = [bbox[1], bbox[0], bbox[3], bbox[2]]
        self.xyxy = not self.xyxy
    
    def get_xyxy(self) -> List[float]:
        if not self.xyxy:
            self.flip_coordinates()
        return self.coordinates
    
    def get_yxyx(self) -> List[float]:
        if self.xyxy:
            self.flip_coordinates()
        return self.coordinates

class ObjectDetectionResult(BaseModel):
    """Result of object detection with bounding box."""

    label: str = Field(..., description="Text description of the detected object")
    box_2d: BoundingBox = Field(..., description="Bounding box as [ymin, xmin, ymax, xmax] in absolute coordinates")

class DetectionResults(BaseModel):
    results: List[ObjectDetectionResult] = Field(default_factory=list, description="List of detected objects")


class ObjectMetadata(BaseModel):
    """Metadata for a single segmented object."""
    
    id: int = Field(..., description="Unique identifier for the object")
    label: str = Field(..., description="Text label/description of the object")
    bbox: List[float] = Field(..., description="Bounding box as [x1, y1, x2, y2]")
    score: Optional[float] = Field(None, description="Confidence score for the segmentation")
    mask_path: Optional[str] = Field(None, description="Path to the saved mask image")
    method: str = Field(..., description="Segmentation method used (e.g., 'bounding_box')")
    status: Optional[str] = Field(None, description="Status of the segmentation process")

    def __hash__(self):
        return hash((self.id, self.label, self.score, self.mask_path, self.method, self.status))
    
    def __eq__(self, other):
        if not isinstance(other, ObjectMetadata):
            return False
        return (self.id, self.label, self.score, self.mask_path, self.method, self.status) == \
               (other.id, other.label, other.score, other.mask_path, other.method, other.status)
    
    @property
    def safe_label(self) -> str:
        """Return a filesystem-safe version of the label."""
        return "".join(
            c for c in self.label if c.isalnum() or c in (" ", "-", "_")
        ).strip().replace(" ", "__")

class SegmentationResults(BaseModel):
    """Complete metadata for a segmentation run."""
    
    image_path: str = Field(..., description="Path to the input image")
    detection_results_path: str = Field(..., description="Path to the detection results JSON file")
    total_objects: int = Field(..., description="Total number of objects to segment")
    output_dir: str = Field(..., description="Directory where outputs are saved")
    objects: List[ObjectMetadata] = Field(default_factory=list, description="List of segmented objects")

    # add validation that all object ids are unique with field validator
    @field_validator('objects')
    @classmethod
    def check_unique_object_ids(cls, v):
        ids = [obj.id for obj in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Object IDs must be unique")
        return v
    
    def get_successful_objects(self) -> List[ObjectMetadata]:
        """Get all objects that were successfully segmented."""
        return [obj for obj in self.objects if obj.status == "success"]
    
    def get_failed_objects(self) -> List[ObjectMetadata]:
        """Get all objects that failed segmentation."""
        return [obj for obj in self.objects if obj.status != "success"]
    
    def success_rate(self) -> float:
        """Calculate the success rate of segmentation."""
        if not self.objects:
            return 0.0
        return len(self.get_successful_objects()) / len(self.objects)


class OClearObject(ObjectMetadata):
    """Metadata for a single object in ObjectClear results."""
    
    removed_object_path: Optional[str] = Field(None, description="Path to the image with the object removed")
    attn_map_path: Optional[str] = Field(None, description="Path to the attention map image")
    segmentation_wo_object_results_path: Optional[str] = Field(None, description="Path to segmentation results without the object")

    def __lt__(self, other):
        if not isinstance(other, OClearObject):
            raise NotImplementedError("Comparison not implemented for different types")
        if (self.segmentation_wo_object_results_path is None) or (other.segmentation_wo_object_results_path is None):
            raise ValueError("`segmentation_wo_object_results_path` is None for self or other")
        
        self_mask = np.array(Image.open(self.mask_path).convert("L"))
        other_mask = np.array(Image.open(other.mask_path).convert("L"))

        overlap = np.logical_and(self_mask, other_mask).sum()
        # if overlap == 0:
        #     return False  # no occlusion, arbitrary order

        with open(self.segmentation_wo_object_results_path, "r") as f:
            self_wo_segmentation_results = ObjectClearResults.model_validate_json(f.read())
        
        with open(other.segmentation_wo_object_results_path, "r") as f:
            other_wo_segmentation_results = ObjectClearResults.model_validate_json(f.read())
        
        for obj in self_wo_segmentation_results.objects:
            if obj.id == other.id:
                other_wo_self_mask_path = obj.mask_path
                other_wo_self_mask = np.array(Image.open(other_wo_self_mask_path).convert("L"))

        for obj in other_wo_segmentation_results.objects:
            if obj.id == self.id:
                self_wo_other_mask_path = obj.mask_path
                self_wo_other_mask = np.array(Image.open(self_wo_other_mask_path).convert("L"))
        
        self_occlusion_by_other = np.logical_and(other_mask, self_wo_other_mask)  # if bigger, self occluded by other
        self_occlusion_by_other = np.logical_and(self_occlusion_by_other, ~self_mask).sum()
        other_occlusion_by_self = np.logical_and(self_mask, other_wo_self_mask)  # if bigger, other occluded by self
        other_occlusion_by_self = np.logical_and(other_occlusion_by_self, ~other_mask).sum()

        return self_occlusion_by_other < other_occlusion_by_self


class ObjectClearResults(SegmentationResults):
    objects: List[OClearObject] = Field(default_factory=list, description="List of segmented objects with ObjectClear metadata")