#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable


DEFAULT_METRICS_FILENAME = "metrics_by_dataset.md"
DEFAULT_NEGATIVE_FILENAME = "negative_cases_by_dataset.md"
GENERATIONS_FILENAME = "generations.jsonl"
REASON_CAN_NOT_EXTRACT = "can_not_extract"


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


def _to_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return False


def _avg_output_token(rows: list[Dict[str, Any]]) -> float | None:
    values = [
        token
        for token in (_to_float(row.get("output_token_length")) for row in rows)
        if token is not None
    ]
    if not values:
        return None
    return mean(values)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _value_label(value: Any) -> str:
    if value is None:
        return "<missing>"
    return str(value)


def _markdown_cell(value: Any) -> str:
    text = _value_label(value)
    return text.replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _negative_extracted_answer(row: Dict[str, Any]) -> Any:
    if row.get("reason") == REASON_CAN_NOT_EXTRACT:
        return None
    return row.get("extracted_answer")


def _has_human_check_answer(row: Dict[str, Any]) -> bool:
    extracted_answer = _negative_extracted_answer(row)
    return extracted_answer is not None and str(extracted_answer).strip() != ""


def _markdown_nullable_cell(value: Any) -> str:
    if value is None:
        return "null"
    return _markdown_cell(value)


def _build_generations_by_index(generations_path: Path) -> dict[str, list[Dict[str, Any]]]:
    by_index: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(_iter_jsonl(generations_path), start=1):
        index = row.get("index")
        if index is None:
            raise ValueError(f"Missing index in generations at {generations_path}:{row_number}")
        key = str(index)
        by_index[key].append(row)
    return by_index


def _enrich_prediction_rows(
    prediction_rows: list[Dict[str, Any]],
    generations_path: Path | None,
) -> list[Dict[str, Any]]:
    generations_by_index: dict[str, list[Dict[str, Any]]] = {}
    if generations_path is not None and generations_path.exists():
        generations_by_index = _build_generations_by_index(generations_path)

    enriched = []
    index_cursor: dict[str, int] = defaultdict(int)
    for row in prediction_rows:
        index = row.get("index")
        gen_row = None
        if index is not None:
            key = str(index)
            candidates = generations_by_index.get(key, [])
            cursor = index_cursor[key]
            pred_dataset = _to_text(row.get("dataset"), "")

            if pred_dataset:
                for pos in range(cursor, len(candidates)):
                    if _to_text(candidates[pos].get("dataset"), "") == pred_dataset:
                        gen_row = candidates[pos]
                        index_cursor[key] = pos + 1
                        break

            if gen_row is None and cursor < len(candidates):
                gen_row = candidates[cursor]
                index_cursor[key] = cursor + 1

            if gen_row is None and candidates:
                gen_row = candidates[-1]

        dataset = _to_text(row.get("dataset"), _to_text((gen_row or {}).get("dataset"), "unknown"))
        question = _to_text(row.get("question"), _to_text((gen_row or {}).get("question"), ""))
        model_output = _to_text(
            row.get("model_output"),
            _to_text((gen_row or {}).get("model_output"), ""),
        )
        enriched.append(
            {
                **row,
                "dataset": dataset,
                "question": question,
                "model_output": model_output,
                "_is_correct": _is_correct(row.get("is_correct")),
            }
        )
    return enriched


def _finish_reason_rows(dataset_rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        groups[_value_label(row.get("finish_reason"))].append(row)

    keys = list(groups.keys())
    ordered = []
    for fixed in ("stop", "length"):
        if fixed in groups:
            ordered.append(fixed)
    ordered.extend(sorted(key for key in keys if key not in {"stop", "length"}))

    rows = []
    for reason in ordered:
        reason_rows = groups[reason]
        correct = sum(1 for row in reason_rows if row["_is_correct"])
        incorrect = len(reason_rows) - correct
        rows.append(
            {
                "finish_reason": reason,
                "count": len(reason_rows),
                "correct": correct,
                "incorrect": incorrect,
                "avg_output_token": _avg_output_token(reason_rows),
            }
        )
    return rows


def _build_metrics_markdown(
    prediction_path: Path,
    generations_path: Path | None,
    rows: list[Dict[str, Any]],
) -> str:
    lines = [
        "# Metrics By Dataset",
        "",
        f"- Prediction: `{prediction_path}`",
    ]
    if generations_path is not None:
        lines.append(f"- Generations: `{generations_path}`")
    lines.append(f"- Total rows: {len(rows)}")
    lines.append("")

    all_correct = sum(1 for row in rows if row["_is_correct"])
    all_incorrect = len(rows) - all_correct
    all_acc = (all_correct / len(rows) * 100.0) if rows else 0.0
    lines.extend(
        [
            "## Overall",
            "",
            f"- total: {len(rows)}",
            f"- correct: {all_correct} ({all_acc:.2f}%)",
            f"- incorrect: {all_incorrect} ({100.0 - all_acc:.2f}%)",
            f"- avg_output_token (correct): {_fmt_float(_avg_output_token([r for r in rows if r['_is_correct']]))}",
            f"- avg_output_token (incorrect): {_fmt_float(_avg_output_token([r for r in rows if not r['_is_correct']]))}",
            "",
        ]
    )

    by_dataset: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        correct_rows = [row for row in dataset_rows if row["_is_correct"]]
        incorrect_rows = [row for row in dataset_rows if not row["_is_correct"]]
        acc = (len(correct_rows) / len(dataset_rows) * 100.0) if dataset_rows else 0.0
        lines.extend(
            [
                f"## Dataset: {dataset}",
                "",
                f"- total: {len(dataset_rows)}",
                f"- correct: {len(correct_rows)} ({acc:.2f}%)",
                f"- incorrect: {len(incorrect_rows)} ({100.0 - acc:.2f}%)",
                f"- avg_output_token (correct): {_fmt_float(_avg_output_token(correct_rows))}",
                f"- avg_output_token (incorrect): {_fmt_float(_avg_output_token(incorrect_rows))}",
                "",
                "### Finish Reason",
                "",
                "| finish_reason | count | correct | incorrect | avg_output_token |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for finish_row in _finish_reason_rows(dataset_rows):
            lines.append(
                f"| {finish_row['finish_reason']} | {finish_row['count']} | {finish_row['correct']} | {finish_row['incorrect']} | {_fmt_float(finish_row['avg_output_token'])} |"
            )

        think_counter = Counter(_value_label(row.get("think_type")) for row in dataset_rows)
        box_counter = Counter(_value_label(row.get("last_box_source")) for row in dataset_rows)

        lines.extend(
            [
                "",
                "### Think Type (count)",
                "",
                "| think_type | count |",
                "| --- | ---: |",
            ]
        )
        for think_type, count in think_counter.most_common():
            lines.append(f"| {think_type} | {count} |")

        lines.extend(
            [
                "",
                "### Last Box Source (count)",
                "",
                "| last_box_source | count |",
                "| --- | ---: |",
            ]
        )
        for source, count in box_counter.most_common():
            lines.append(f"| {source} | {count} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_negative_markdown(rows: list[Dict[str, Any]]) -> str:
    by_dataset: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["_is_correct"] and _has_human_check_answer(row):
            by_dataset[row["dataset"]].append(row)

    total_negative = sum(len(items) for items in by_dataset.values())
    lines = [
        "# Negative Cases By Dataset",
        "",
        f"- Total human-check negative rows: {total_negative}",
        "",
    ]

    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        lines.extend(
            [
                f"## Dataset: {dataset}",
                "",
                f"- human-check negative rows: {len(dataset_rows)}",
                "",
                "| dataset | index | label | extracted_answer |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for row in dataset_rows:
            lines.append(
                f"| {_markdown_cell(row.get('dataset'))} | {_markdown_cell(row.get('index'))} | "
                f"{_markdown_cell(row.get('label'))} | "
                f"{_markdown_nullable_cell(_negative_extracted_answer(row))} |"
            )
        lines.append("")

    if total_negative == 0:
        lines.append("No human-check negative rows found.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset-level benchmark metrics.")
    parser.add_argument("--prediction", required=True, help="Path to prediction.jsonl")
    parser.add_argument(
        "--generations",
        help="Path to generations.jsonl. Defaults to sibling generations.jsonl.",
    )
    parser.add_argument(
        "--metrics-output",
        default=DEFAULT_METRICS_FILENAME,
        help=f"Output metrics markdown filename (default: {DEFAULT_METRICS_FILENAME})",
    )
    parser.add_argument(
        "--negative-output",
        default=None,
        help=f"Output negative markdown filename (default: {DEFAULT_NEGATIVE_FILENAME})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    if args.generations:
        generations_path = Path(args.generations)
    else:
        generations_path = prediction_path.with_name(GENERATIONS_FILENAME)
    if not generations_path.exists():
        generations_path = None

    prediction_rows = list(_iter_jsonl(prediction_path))
    rows = _enrich_prediction_rows(prediction_rows, generations_path)

    metrics_text = _build_metrics_markdown(prediction_path, generations_path, rows)
    negative_text = _build_negative_markdown(rows)

    output_dir = prediction_path.parent
    metrics_output_path = output_dir / args.metrics_output
    negative_output_name = args.negative_output or DEFAULT_NEGATIVE_FILENAME
    negative_output_path = output_dir / negative_output_name
    metrics_output_path.write_text(metrics_text, encoding="utf-8")
    negative_output_path.write_text(negative_text, encoding="utf-8")

    print(f"Prediction: {prediction_path}")
    print(f"Generations: {generations_path if generations_path is not None else '<not found>'}")
    print(f"Rows: {len(rows)}")
    print(f"Metrics output: {metrics_output_path}")
    print(f"Negative output: {negative_output_path}")


if __name__ == "__main__":
    main()
