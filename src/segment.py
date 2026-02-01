import os

# --- Imports from SAM 3 Repository ---
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

from .utils import (
    SamPredictedSegmentationMask,
    SamSegmentationResult,
    add_visual_prompt,
    create_empty_datapoint,
    is_mask_unique,
    process_raw_segmentation_results,
    save_mask_to_dir,
    set_image,
)
from .models import ObjectMetadata, SegmentationResults, DetectionResults


SEGMENTATION_RESULTS_FILENAME = "segmentation_results.json"


class SAM3Segmenter:
    """SAM3-based image segmenter with configurable parameters."""

    def __init__(self, detection_threshold: float = 0.3):
        """
        Initialize SAM3 segmenter.

        Args:
            detection_threshold: Minimum confidence threshold for detections
            image_size: Target image size for processing
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Setup device
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load Model
        self.model = build_sam3_image_model()
        self.model.to(self.device)

        # Transforms
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=1008, max_size=1008, square=True, consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Post-processor (Configured to return Boxes + Masks)
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=True,  # Move results to CPU for easy viewing
        )

    def segment_objects(
        self,
        image: Image.Image,
        prompts: list[str],
        bboxes: list[list[float]],
        textual_prompts_only: bool,
    ) -> list[SamSegmentationResult]:
        # Setup datapoint
        datapoint = create_empty_datapoint()
        set_image(datapoint, image)

        # Add all visual prompts
        query_ids = []
        for i, prompt in enumerate(prompts):

            query_id = add_visual_prompt(
                datapoint,
                boxes=None if textual_prompts_only else [bboxes[i]],
                points=None,
                prompt=prompt,
            )
            query_ids.append(query_id)

        # Prepare batch
        datapoint_transformed = self.transform(datapoint)
        batch = collate([datapoint_transformed], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        # Run model
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(batch)

        # Process results
        processed_results = self.postprocessor.process_results(
            output, batch.find_metadatas
        )

        # Return results mapped by original order
        raw_segmentation_results = [processed_results[i] for i in query_ids]
        segmentation_results = process_raw_segmentation_results(
            raw_results=raw_segmentation_results,
            prompts=prompts,
            bboxes=bboxes,
            image_size=image.size,
        )
        return segmentation_results

    def filter_and_select_masks(
        self, segmentation_results: list[SamSegmentationResult], output_dir: str, method: str
    ) -> tuple[list, list[ObjectMetadata]]:
        chosen_objects_metadata = []
        chosen_masks = []

        # Process results for each detected object and save first pass
        for i, seg_res in enumerate(segmentation_results):
            print(f"[Object {i + 1}/{len(segmentation_results)}] Found {len(seg_res)} mask segments for '{seg_res.prompt}'")

            object_metadata = ObjectMetadata(
                id=i + 1,
                label=seg_res.prompt,
                bbox=seg_res.bbox,
                score=None,
                mask_path=None,
                method=method,
                status=None,
            )
            chosen_objects_metadata.append(object_metadata)

            filtered_masks = [
                seg for seg in seg_res.segmentation_masks if seg.bbox_overlap >= 0.5
            ]
            if len(filtered_masks) == 0:
                print(f"No segments found for '{seg_res.prompt}'")
                # Add failed first pass to metadata
                object_metadata.status = "no_segments_found"
                continue

            filtered_unique_masks: list[SamPredictedSegmentationMask] = []
            for mask_result in filtered_masks:
                if is_mask_unique(mask_result.mask, chosen_masks, threshold=0.9):
                    filtered_unique_masks.append(mask_result)

            if len(filtered_unique_masks) == 0:
                print(f"No segments found for '{seg_res.prompt}'")
                # Add failed first pass to metadata
                object_metadata.status = "rejected_not_unique"
                continue

            best_segmentation = max(filtered_unique_masks, key=lambda seg: seg.score)
            best_mask = best_segmentation.mask

            chosen_masks.append(best_mask)

            # add the seg bounding box to mask:
            import numpy as np
            from .utils import get_bbox_mask
            bbox_mask = get_bbox_mask(seg_res.bbox, seg_res.image_size)

            step = 2
            bbox_left_frame = (bbox_mask & (~np.roll(bbox_mask, step, axis=1)))
            bbox_right_frame = (bbox_mask & (~np.roll(bbox_mask, -step, axis=1)))
            bbox_top_frame = (bbox_mask & (~np.roll(bbox_mask, step, axis=0)))
            bbox_bottom_frame = (bbox_mask & (~np.roll(bbox_mask, -step, axis=0)))
            bbox_frame = bbox_left_frame | bbox_right_frame | bbox_top_frame | bbox_bottom_frame
            
            framed_mask = best_mask | bbox_frame
            save_mask_to_dir(
                framed_mask,
                f"framed_segmask_{i + 1:02d}",
                seg_res.prompt,
                best_segmentation.score,
                output_dir,
            )

            saved_path = save_mask_to_dir(
                best_mask,
                f"segmask_{i + 1:02d}",
                seg_res.prompt,
                best_segmentation.score,
                output_dir,
            )

            object_metadata.score = float(best_segmentation.score)
            object_metadata.mask_path = saved_path
            object_metadata.status = "success"

        return chosen_masks, chosen_objects_metadata

    def process_layered_segmentation(
        self, image_path: str, detection_path: str, output_dir: str = "output"
    ):
        """
        Process layered segmentation with first and second pass segmentations.

        Args:
            image_path: Path to the input image
            detection_data: Either a path to JSON file with detections or list of detection results
            output_dir: Directory to save masks and metadata

        Returns:
            Dictionary containing segmentation metadata and results
        """
        image = Image.open(image_path)

        with open(detection_path, "r") as f:
            detection_data = DetectionResults.model_validate_json(f.read())

        print(f"=== Segmenting {len(detection_data.results)} detected objects from {image_path} ===")

        prompts = [obj.label for obj in detection_data.results]
        bboxes = [obj.box_2d.get_xyxy() for obj in detection_data.results]

        # === FIRST PASS: Segment all detected objects at once ===
        print("\n=== First Pass Results (Bounding Box Based) ===")
        segmentation_results = self.segment_objects(
            image=image, prompts=prompts, bboxes=bboxes, textual_prompts_only=True
        )

        segmentation_results_fallback = self.segment_objects(
            image=image, prompts=prompts, bboxes=bboxes, textual_prompts_only=False
        )

        first_pass_output_dir = os.path.join(output_dir, "text_prompt")
        chosen_masks, objects_metadata_text = self.filter_and_select_masks(
            segmentation_results, first_pass_output_dir, method="text prompt"
        )

        second_pass_output_dir = os.path.join(output_dir, "bbox_prompt")
        chosen_masks_fallback, objects_metadata_bbox = self.filter_and_select_masks(
            segmentation_results_fallback, second_pass_output_dir, method="text prompt + bbox"
        )

        objects_metadata = []
        for i in range(len(objects_metadata_text)):
            if objects_metadata_text[i].status == "success":
                objects_metadata.append(objects_metadata_text[i])
            else:
                objects_metadata.append(objects_metadata_bbox[i])


        # Initialize metadata collection
        segmentation_results = SegmentationResults(
            image_path=image_path,
            detection_results_path=detection_path,
            total_objects=len(detection_data.results),
            output_dir=output_dir,
            objects=objects_metadata,
        )

        segmentation_results_path = os.path.join(
            output_dir, SEGMENTATION_RESULTS_FILENAME
        )

        os.makedirs(output_dir, exist_ok=True)
        with open(segmentation_results_path, "w") as f:
            f.write(segmentation_results.model_dump_json(indent=2))

        print("\n=== Summary ===")
        print(f"First pass (bbox-based): Processed {len(detection_data.results)} objects")
        print(f"Metadata saved to: {segmentation_results_path}")
        print(f"Check {os.path.join(output_dir, 'masks')}/ directory for saved masks")

        # Calculate detailed statistics
        total_objects = len(detection_data.results)
        first_pass_successes = len(segmentation_results.get_successful_objects())
        first_pass_failures = len(segmentation_results.get_failed_objects())
        success_rate = segmentation_results.success_rate()

        print("\n=== Detailed Statistics ===")
        print(f"Original objects detected: {total_objects}")
        print(
            f"First pass successes: {first_pass_successes}/{total_objects} ({success_rate:.1%})"
        )
        print(
            f"First pass failures: {first_pass_failures}/{total_objects} ({first_pass_failures / total_objects:.1%})"
        )


if __name__ == "__main__":
    image_path = os.getenv("IMAGE_PATH")
    detection_path = os.getenv("DETECTION_PATH")
    output_dir = os.getenv("OUTPUT_DIR")

    assert os.path.isfile(image_path), f"Image file not found: {image_path}"
    assert os.path.isfile(detection_path), f"Detection results file not found: {detection_path}"

    segmenter = SAM3Segmenter()
    segmenter.process_layered_segmentation(
        image_path=image_path,
        detection_path=detection_path,
        output_dir=output_dir,
    )
