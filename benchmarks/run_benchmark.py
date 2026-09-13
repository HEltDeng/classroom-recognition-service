from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2

from src.core.config import ServiceSettings
from src.omr.template import TemplateRepository
from src.preprocessing.crops import extract_text_crop
from src.preprocessing.geometry import preprocess_page
from src.providers.registry import ProviderRegistry


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(prediction: str | None, truth: str) -> float:
    if not truth:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction or "", truth) / len(truth)


def _resource_snapshot() -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info().rss
        cpu = psutil.cpu_percent(interval=None)
        return {"process_rss_mb": round(memory / 1024**2, 2), "cpu_percent": cpu}
    except ImportError:
        return {"resource_metrics": "psutil_not_installed"}


def _run_one(provider, template, processed, question_no: int, truth: str, input_path: Path) -> dict[str, Any]:
    region = next(
        item for item in template.text_regions if item.question_no == question_no
    )
    crop = extract_text_crop(processed.warped_gray, region)
    started = time.perf_counter()
    result = provider.recognize_crop(crop)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "provider": provider.name,
        "model": provider.model_version,
        "template_version": template.template_version,
        "input": str(input_path),
        "question_no": question_no,
        "truth": truth,
        "raw_text": result.raw_text,
        "status": result.status.value,
        "confidence": result.confidence,
        "exact_match": result.raw_text == truth,
        "cer": round(character_error_rate(result.raw_text, truth), 4),
        "latency_ms": elapsed_ms,
        "available": provider.is_available,
        "metadata": result.metadata,
        "resources": _resource_snapshot(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    templates = TemplateRepository(Path(args.root) / "templates")
    template = templates.load(args.template_version)
    registry = ProviderRegistry(ServiceSettings.from_env())
    provider = registry.get(args.provider)
    image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {args.input}")
    processed = preprocess_page(image, template)
    if processed.warped_gray is None or processed.errors:
        raise SystemExit(f"Preprocessing failed: {processed.errors}")
    if args.questions:
        question_numbers = [int(value.strip()) for value in args.questions.split(",") if value.strip()]
    else:
        question_numbers = [args.question]
    if args.truths:
        truths = [value.strip() for value in args.truths.split(",")]
    else:
        truths = [args.truth or ""]
    if len(truths) == 1 and len(question_numbers) > 1:
        truths *= len(question_numbers)
    if len(truths) != len(question_numbers):
        raise SystemExit("--truths must contain one value per --questions item")

    samples = [
        _run_one(provider, template, processed, question_no, truth, args.input)
        for question_no, truth in zip(question_numbers, truths)
    ]
    if len(samples) == 1:
        return samples[0]
    exact_count = sum(1 for sample in samples if sample["exact_match"])
    return {
        "provider": provider.name,
        "model": provider.model_version,
        "template_version": template.template_version,
        "input": str(args.input),
        "samples": samples,
        "summary": {
            "count": len(samples),
            "exact_match_count": exact_count,
            "exact_match_rate": round(exact_count / len(samples), 4),
            "mean_cer": round(sum(sample["cer"] for sample in samples) / len(samples), 4),
            "mean_latency_ms": round(sum(sample["latency_ms"] for sample in samples) / len(samples), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one provider on one fixed OCR crop")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--truth")
    parser.add_argument("--question", type=int, default=5)
    parser.add_argument("--truths", help="Comma-separated truths for --questions")
    parser.add_argument("--questions", help="Comma-separated text question numbers")
    parser.add_argument("--template-version", default="A-v0.1")
    parser.add_argument("--provider", choices=["paddleocr", "trocr", "deepseek_ocr2", "deepseek_vision"], default="paddleocr")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
