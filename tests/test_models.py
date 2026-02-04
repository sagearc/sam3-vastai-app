"""Tests for Pydantic models."""

from pathlib import Path
from src.models import DetectionResults, BoundingBox, ObjectDetectionResult


EXAMPLE_DIR = Path(__file__).parent.parent / "example"


class TestDetectionResults:
    """Tests for DetectionResults model."""

    def test_load_office_detection_results(self):
        """Test loading office_detection_results.json as DetectionResults."""
        json_path = EXAMPLE_DIR / "office_detection_results.json"
        assert json_path.exists(), f"Example file not found: {json_path}"

        with open(json_path, "r") as f:
            detection_results = DetectionResults.model_validate_json(f.read())

        # Verify it loaded correctly
        assert isinstance(detection_results, DetectionResults)
        assert len(detection_results.results) > 0

        # Verify first result structure
        first_result = detection_results.results[0]
        assert isinstance(first_result, ObjectDetectionResult)
        assert isinstance(first_result.label, str)
        assert len(first_result.label) > 0
        assert isinstance(first_result.box_2d, BoundingBox)
        assert len(first_result.box_2d.coordinates) == 4
