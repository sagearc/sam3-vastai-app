import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.models import DetectionResults, SegmentationResults
from src.segment import SAM3Segmenter


# Global segmenter instance (loaded once at startup)
segmenter: SAM3Segmenter | None = None

# Output directory for artifacts
OUTPUT_BASE_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SAM3 model at startup."""
    global segmenter
    print("Loading SAM3 model...")
    segmenter = SAM3Segmenter()
    print("SAM3 model loaded successfully!")
    
    # Ensure output directory exists
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup on shutdown (optional)
    print("Shutting down...")


app = FastAPI(
    title="SAM3 Segmentation API",
    description="API for image segmentation using SAM3 model",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files for serving artifacts
app.mount("/artifacts", StaticFiles(directory=str(OUTPUT_BASE_DIR)), name="artifacts")


class SegmentationResponse(SegmentationResults):
    """Response model with artifact URLs."""
    artifacts_base_url: str


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    image: Annotated[UploadFile, File(description="Input image file")],
    detection_results: Annotated[str, Form(description="Detection results as JSON string")],
):
    """
    Segment objects in an image based on detection results.
    
    Args:
        image: The input image file (PNG, JPG, etc.)
        detection_results: JSON string containing detection results
        
    Returns:
        Segmentation results with paths to generated mask artifacts
    """
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Validate detection results
    try:
        detection_data = DetectionResults.model_validate_json(detection_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid detection results: {e}")
    
    # Create unique output directory for this request
    request_id = str(uuid.uuid4())[:8]
    output_dir = OUTPUT_BASE_DIR / request_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded image temporarily
    image_path = output_dir / f"input_{image.filename}"
    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
    finally:
        await image.close()
    
    # Save detection results
    detection_path = output_dir / "detection_results.json"
    with open(detection_path, "w") as f:
        f.write(detection_results)
    
    try:
        # Run segmentation
        segmenter.process_layered_segmentation(
            image_path=str(image_path),
            detection_path=str(detection_path),
            output_dir=str(output_dir),
        )
        
        # Load the generated results
        results_path = output_dir / "segmentation_results.json"
        if not results_path.exists():
            raise HTTPException(status_code=500, detail="Segmentation failed to produce results")
        
        with open(results_path, "r") as f:
            seg_results = SegmentationResults.model_validate_json(f.read())
        
        # Convert absolute paths to relative paths for the response
        relative_results = convert_to_relative_paths(seg_results, output_dir, request_id)
        
        return SegmentationResponse(
            **relative_results.model_dump(),
            artifacts_base_url=f"/artifacts/{request_id}",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")


def convert_to_relative_paths(
    results: SegmentationResults,
    output_dir: Path,
    request_id: str,
) -> SegmentationResults:
    """Convert absolute paths in results to relative paths."""
    # Update image path to relative
    results.image_path = f"{request_id}/input_{Path(results.image_path).name}"
    results.detection_results_path = f"{request_id}/detection_results.json"
    results.output_dir = request_id
    
    # Update mask paths to relative
    for obj in results.objects:
        if obj.mask_path:
            # Convert absolute path to relative path from output base
            mask_path = Path(obj.mask_path)
            obj.mask_path = str(mask_path.relative_to(OUTPUT_BASE_DIR))
    
    return results


@app.get("/artifacts/{request_id}/masks/{filename}")
async def get_mask(request_id: str, filename: str):
    """Retrieve a specific mask artifact."""
    mask_path = OUTPUT_BASE_DIR / request_id / "masks" / filename
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Mask not found")
    return FileResponse(mask_path, media_type="image/png")


@app.get("/health")
async def health_check():
    """Check if the service is healthy and model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": segmenter is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
