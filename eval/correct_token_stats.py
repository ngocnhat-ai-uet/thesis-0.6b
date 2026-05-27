#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


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


def _is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return False


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def build_correct_token_stats(prediction_path: Path) -> str:
    rows_by_dataset: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(prediction_path):
        if _is_correct(row.get("is_correct")):
            dataset = str(row.get("dataset") or "unknown")
            rows_by_dataset[dataset].append(row)

    lines = [
        f"Prediction: {prediction_path}",
        "",
        "| dataset | correct | q1 | q2 | q3 | p90 | p95 | <4096 | 4096-8192 | >4096 | >8192 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dataset in sorted(rows_by_dataset):
        rows = rows_by_dataset[dataset]
        tokens = sorted(
            token
            for token in (_to_float(row.get("output_token_length")) for row in rows)
            if token is not None
        )
        under_4096 = sum(1 for token in tokens if token < 4096)
        between_4096_8192 = sum(1 for token in tokens if 4096 <= token <= 8192)
        over_4096 = sum(1 for token in tokens if token > 4096)
        over_8192 = sum(1 for token in tokens if token > 8192)
        lines.append(
            f"| {dataset} | {len(rows)} | "
            f"{_fmt(_percentile(tokens, 0.25))} | "
            f"{_fmt(_percentile(tokens, 0.50))} | "
            f"{_fmt(_percentile(tokens, 0.75))} | "
            f"{_fmt(_percentile(tokens, 0.90))} | "
            f"{_fmt(_percentile(tokens, 0.95))} | "
            f"{under_4096} | {between_4096_8192} | "
            f"{over_4096} | {over_8192} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print output-token statistics for correct predictions by dataset."
    )
    parser.add_argument("--prediction", required=True, help="Path to prediction.jsonl")
    parser.add_argument(
        "--output",
        help="Path to output markdown file. Defaults to correct_token_stats.md next to prediction.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    output_path = (
        Path(args.output)
        if args.output
        else prediction_path.with_name("correct_token_stats.md")
    )
    text = build_correct_token_stats(prediction_path)
    output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
