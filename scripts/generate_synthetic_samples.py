from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow the script to be run directly from the repository root or from any
# working directory without requiring a package installation first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.synthetic_factory import build_sheet


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        (
            "answer-sheet-a-v0.1-recognized.png",
            build_sheet({1: "B", 2: "C", 3: "A", 4: "D"}, perspective=True),
            {"1": "B", "2": "C", "3": "A", "4": "D"},
        ),
        ("answer-sheet-a-v0.1-blank.png", build_sheet(perspective=True), {}),
        (
            "answer-sheet-a-v0.1-multiple.png",
            build_sheet(multiple={1: ("A", "C")}, perspective=True),
            {"1": "MULTIPLE"},
        ),
        (
            "answer-sheet-a-v0.1-too-light.png",
            build_sheet(light={1: "D"}, perspective=True),
            {"1": "TOO_LIGHT"},
        ),
        (
            "answer-sheet-a-v0.1-ambiguous.png",
            build_sheet(ambiguous={1: ("A", "B")}, perspective=True),
            {"1": "AMBIGUOUS"},
        ),
        (
            "answer-sheet-a-v0.1-ocr-synthetic.png",
            build_sheet(
                answers={1: "B", 2: "C", 3: "A", 4: "D"},
                text_values={5: "because", 6: "environment", 7: "went"},
                perspective=True,
            ),
            {"5": "because", "6": "environment", "7": "went"},
        ),
        (
            "answer-sheet-a-v0.1-misspelling.png",
            build_sheet(text_values={5: "becuse"}, perspective=True),
            {"5": "becuse"},
        ),
    ]
    manifest = []
    for filename, image, labels in samples:
        path = output_dir / filename
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")
        manifest.append({"file": filename, "ground_truth": labels, "pii": False})
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate non-PII synthetic answer-sheet fixtures")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "synthetic",
    )
    generate(parser.parse_args().output)
