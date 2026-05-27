#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


GENERATIONS_FILENAME = "generations.jsonl"
SUSPECT_FALSE_NEGATIVE_DIRNAME = "suspect_false_negative"
CHUNK_SIZE = 100
REASON_CAN_NOT_EXTRACT = "can_not_extract"
REASON_NO_MATCH = "no_match"
LAST_BOX_SOURCE_SOLUTION = "solution"
LAST_BOX_SOURCE_THOUGHT = "thought"
THINK_CLOSE = "</think>"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")

            yield row


def _is_incorrect(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false"}
    return False


def _require_index(row: Dict[str, Any], path: Path, row_number: int) -> Any:
    if "index" not in row:
        raise ValueError(f"Missing index at {path}:{row_number}")
    return row["index"]


def _review_model_output(prediction: Dict[str, Any], model_output: Any) -> str:
    text = "" if model_output is None else str(model_output)
    reason = prediction.get("reason")
    last_box_source = prediction.get("last_box_source")

    if reason == REASON_CAN_NOT_EXTRACT:
        return text

    if reason != REASON_NO_MATCH:
        return text

    close_idx = text.find(THINK_CLOSE)

    if last_box_source == LAST_BOX_SOURCE_SOLUTION:
        if close_idx == -1:
            return text
        return text[close_idx + len(THINK_CLOSE) :].strip()

    if last_box_source == LAST_BOX_SOURCE_THOUGHT:
        if close_idx == -1:
            return text
        return text[:close_idx].strip()

    return text


def load_generations_by_index(generations_path: Path) -> Dict[str, Dict[str, Any]]:
    generations_by_index: Dict[str, Dict[str, Any]] = {}
    duplicate_indexes = []

    for row_number, row in enumerate(_iter_jsonl(generations_path), start=1):
        index = str(_require_index(row, generations_path, row_number))
        if index in generations_by_index:
            duplicate_indexes.append(index)
            continue
        generations_by_index[index] = row

    if duplicate_indexes:
        examples = duplicate_indexes[:5]
        raise ValueError(
            f"{generations_path} contains duplicate index values. Examples: {examples}"
        )

    return generations_by_index


def export_suspect_false_negative(
    prediction_path: Path,
    generations_path: Path | None = None,
    output_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    if generations_path is None:
        generations_path = prediction_path.with_name(GENERATIONS_FILENAME)
    if output_dir is None:
        output_dir = prediction_path.with_name(SUSPECT_FALSE_NEGATIVE_DIRNAME)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    generations_by_index = load_generations_by_index(generations_path)

    total = 0
    missing_generation_indexes = []
    output_rows = []

    for row_number, prediction in enumerate(_iter_jsonl(prediction_path), start=1):
        total += 1
        index = str(_require_index(prediction, prediction_path, row_number))

        if not _is_incorrect(prediction.get("is_correct")):
            continue

        generation = generations_by_index.get(index)
        if generation is None:
            missing_generation_indexes.append(index)
            continue

        output_rows.append(
            {
                "index": prediction.get("index"),
                "question": generation.get("question", ""),
                "label": prediction.get("label", generation.get("label", "")),
                "model_output": _review_model_output(
                    prediction,
                    generation.get("model_output", ""),
                ),
            }
        )

    if missing_generation_indexes:
        examples = missing_generation_indexes[:5]
        raise ValueError(
            f"{len(missing_generation_indexes)} incorrect predictions have no matching "
            f"generation row. Examples: {examples}"
        )

    write_chunks(output_rows, output_dir, chunk_size)

    print(f"Prediction: {prediction_path}")
    print(f"Generations: {generations_path}")
    print(f"Output: {output_dir}")
    print(f"Prediction rows: {total}")
    print(f"Suspect false-negative rows: {len(output_rows)}")


def write_chunks(records: list[Dict[str, Any]], output_dir: Path, chunk_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("suspect_false_negative_part*_of*_rows*.json"):
        old_path.unlink()

    total_parts = (len(records) + chunk_size - 1) // chunk_size
    row_width = len(str(len(records))) if records else 1

    for part_idx in range(total_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, len(records))
        output_path = output_dir / (
            f"suspect_false_negative_part{part_idx + 1:02d}_of{total_parts:02d}"
            f"_rows{start + 1:0{row_width}d}-{end:0{row_width}d}.json"
        )
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(records[start:end], output_file, ensure_ascii=False, indent=2)
        print(f"wrote {end - start} records -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export incorrect prediction rows for false-negative review."
    )
    parser.add_argument("--prediction", required=True, help="Path to prediction.jsonl")
    parser.add_argument(
        "--generations",
        help="Path to generations.jsonl. Defaults to the file next to prediction.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Path to suspect_false_negative output directory. Defaults to the folder next to "
            "prediction.jsonl."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_suspect_false_negative(
        prediction_path=Path(args.prediction),
        generations_path=Path(args.generations) if args.generations else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
