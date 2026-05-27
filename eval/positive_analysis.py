#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable

import pandas as pd


CONFIG_FILENAME = "config.resolved.yaml"
GENERATIONS_FILENAME = "generations.jsonl"
OUTPUT_FILENAME = "_positive_analysis.txt"
PRIMARY_FINISH_REASONS = ("length", "stop")
RATIO_GROUPS = (
    ("Group 1: ratio < 0.5", None, 0.5),
    ("Group 2: 0.5 <= ratio < 0.8", 0.5, 0.8),
    ("Group 3: 0.8 <= ratio < 1.0", 0.8, 1.0),
    ("Group 4: 1.0 <= ratio < 1.2", 1.0, 1.2),
    ("Group 5: 1.2 <= ratio < 1.5", 1.2, 1.5),
    ("Group 6: 1.5 <= ratio < 2.0", 1.5, 2.0),
    ("Group 7: ratio >= 2.0", 2.0, None),
)


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


def _value_label(value: Any) -> str:
    if value is None:
        return "<missing>"
    return str(value)


def _numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_counter(title: str, counter: Counter[str], indent: str = "") -> list[str]:
    lines = [f"{indent}{title}:"]
    if not counter:
        lines.append(f"{indent}  <none>: 0")
        return lines

    for key, count in counter.most_common():
        lines.append(f"{indent}  {key}: {count}")
    return lines


def _quantiles(values: list[float]) -> tuple[float, float, float] | None:
    if not values:
        return None
    sorted_values = sorted(values)

    def percentile(p: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        pos = (len(sorted_values) - 1) * p
        lower = int(pos)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = pos - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    return percentile(0.25), percentile(0.5), percentile(0.75)


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _load_generations_by_index(generations_path: Path) -> dict[str, Dict[str, Any]]:
    generations_by_index: dict[str, Dict[str, Any]] = {}
    duplicate_indexes = []
    for row_number, row in enumerate(_iter_jsonl(generations_path), start=1):
        if "index" not in row:
            raise ValueError(f"Missing index at {generations_path}:{row_number}")
        index = str(row["index"])
        if index in generations_by_index:
            duplicate_indexes.append(index)
            continue
        generations_by_index[index] = row

    if duplicate_indexes:
        raise ValueError(
            f"{generations_path} contains duplicate index values. "
            f"Examples: {duplicate_indexes[:5]}"
        )

    return generations_by_index


def _load_dataset_path_from_config(config_path: Path) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to infer dataset path from config") from exc

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("data_path"):
        raise ValueError(f"{config_path} does not contain dataset.data_path")

    return Path(dataset["data_path"])


def _load_train_tokens(dataset_path: Path) -> dict[str, float]:
    dataset = pd.read_parquet(dataset_path, columns=["index", "train_token"])
    missing = {"index", "train_token"} - set(dataset.columns)
    if missing:
        raise ValueError(f"{dataset_path} is missing required columns: {sorted(missing)}")

    index_as_text = dataset["index"].astype(str)
    duplicated = index_as_text.duplicated()
    if duplicated.any():
        examples = index_as_text.loc[duplicated].head(5).tolist()
        raise ValueError(f"{dataset_path} contains duplicate index values: {examples}")

    return {
        str(row["index"]): float(row["train_token"])
        for row in dataset[["index", "train_token"]].to_dict("records")
    }


def _resolve_input_paths(
    prediction_path: Path,
    generations_path: Path | None,
    dataset_path: Path | None,
) -> tuple[Path, Path]:
    if generations_path is None:
        generations_path = prediction_path.with_name(GENERATIONS_FILENAME)
    if dataset_path is None:
        dataset_path = _load_dataset_path_from_config(
            prediction_path.with_name(CONFIG_FILENAME)
        )
    return generations_path, dataset_path


def _ratio_group_label(ratio: float) -> str:
    for label, lower, upper in RATIO_GROUPS:
        if lower is not None and ratio < lower:
            continue
        if upper is not None and ratio >= upper:
            continue
        return label
    raise ValueError(f"Unexpected ratio: {ratio}")


def _build_correct_rows(
    prediction_rows: list[Dict[str, Any]],
    generations_by_index: dict[str, Dict[str, Any]],
    train_tokens_by_index: dict[str, float],
) -> tuple[list[Dict[str, Any]], list[str]]:
    correct_rows = []
    missing_indexes = []

    for row in prediction_rows:
        if not _is_correct(row.get("is_correct")):
            continue

        index = str(row.get("index"))
        generation = generations_by_index.get(index)
        target_token = train_tokens_by_index.get(index)
        output_token = None
        if generation is not None:
            output_token = _numeric_value(generation.get("output_token_length"))
        if output_token is None:
            output_token = _numeric_value(row.get("output_token_length"))

        if generation is None or target_token is None:
            missing_indexes.append(index)

        correct_rows.append(
            {
                **row,
                "_output_token": output_token,
                "_target_token": target_token,
            }
        )

    return correct_rows, missing_indexes


def _token_stats(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    output_tokens = [
        token
        for token in (_numeric_value(row.get("_output_token")) for row in rows)
        if token is not None
    ]
    quantiles = _quantiles(output_tokens)
    if quantiles is None:
        return {"count": 0, "avg": None, "q1": None, "q2": None, "q3": None}
    return {
        "count": len(output_tokens),
        "avg": mean(output_tokens),
        "q1": quantiles[0],
        "q2": quantiles[1],
        "q3": quantiles[2],
    }


def _ratio_counts(rows: list[Dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        output_token = _numeric_value(row.get("_output_token"))
        target_token = _numeric_value(row.get("_target_token"))
        if output_token is None or target_token is None or target_token <= 0:
            continue
        counter[_ratio_group_label(output_token / target_token)] += 1
    return counter


def _format_token_stats(title: str, stats: Dict[str, Any], indent: str = "") -> list[str]:
    lines = [f"{indent}{title}:"]
    if stats["count"] == 0:
        lines.append(f"{indent}  count: 0")
        return lines
    lines.extend(
        [
            f"{indent}  count: {stats['count']}",
            f"{indent}  avg output_token: {_format_float(stats['avg'])}",
            f"{indent}  q1: {_format_float(stats['q1'])}",
            f"{indent}  q2: {_format_float(stats['q2'])}",
            f"{indent}  q3: {_format_float(stats['q3'])}",
        ]
    )
    return lines


def _format_ratio_counts(counter: Counter[str], indent: str = "") -> list[str]:
    lines = [f"{indent}Output/target token ratio:"]
    for label, _, _ in RATIO_GROUPS:
        lines.append(f"{indent}  {label}: {counter.get(label, 0)}")
    return lines


def _format_analysis(
    prediction_path: Path,
    generations_path: Path,
    dataset_path: Path,
    prediction_rows: list[Dict[str, Any]],
    correct_rows: list[Dict[str, Any]],
    missing_indexes: list[str],
) -> str:
    finish_counter = Counter(_value_label(row.get("finish_reason")) for row in correct_rows)
    lines = [
        f"Prediction: {prediction_path}",
        f"Generations: {generations_path}",
        f"Dataset: {dataset_path}",
        f"Total rows: {len(prediction_rows)}",
        f"Correct rows: {len(correct_rows)}",
    ]
    lines.extend(_format_counter("Correct by finish_reason", finish_counter))
    lines.append("")
    lines.extend(_format_token_stats("Overall statistic", _token_stats(correct_rows)))
    lines.extend(_format_ratio_counts(_ratio_counts(correct_rows)))

    finish_order = list(PRIMARY_FINISH_REASONS)
    finish_order.extend(
        finish_reason
        for finish_reason in sorted(finish_counter)
        if finish_reason not in PRIMARY_FINISH_REASONS
    )

    for finish_reason in finish_order:
        finish_rows = [
            row
            for row in correct_rows
            if _value_label(row.get("finish_reason")) == finish_reason
        ]
        lines.append("")
        lines.append(f"finish_reason={finish_reason}: {len(finish_rows)}")
        lines.extend(_format_token_stats("output_token", _token_stats(finish_rows), indent="  "))
        lines.extend(
            _format_counter(
                "think_type",
                Counter(_value_label(row.get("think_type")) for row in finish_rows),
                indent="  ",
            )
        )
        lines.extend(
            _format_counter(
                "last_box_source",
                Counter(_value_label(row.get("last_box_source")) for row in finish_rows),
                indent="  ",
            )
        )

    if missing_indexes:
        lines.append("")
        lines.append(
            f"Warning: {len(missing_indexes)} correct rows are missing generation or target token."
        )
        lines.append(f"Examples: {', '.join(missing_indexes[:10])}")

    return "\n".join(lines) + "\n"


def analyze_positive_predictions(
    prediction_path: Path,
    generations_path: Path | None = None,
    dataset_path: Path | None = None,
) -> None:
    generations_path, dataset_path = _resolve_input_paths(
        prediction_path,
        generations_path,
        dataset_path,
    )
    prediction_rows = list(_iter_jsonl(prediction_path))
    generations_by_index = _load_generations_by_index(generations_path)
    train_tokens_by_index = _load_train_tokens(dataset_path)
    correct_rows, missing_indexes = _build_correct_rows(
        prediction_rows,
        generations_by_index,
        train_tokens_by_index,
    )
    analysis_text = _format_analysis(
        prediction_path,
        generations_path,
        dataset_path,
        prediction_rows,
        correct_rows,
        missing_indexes,
    )
    output_path = prediction_path.with_name(OUTPUT_FILENAME)
    output_path.write_text(analysis_text, encoding="utf-8")
    print(analysis_text, end="")
    print()
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correct rows in prediction.jsonl by finish reason and token ratio."
    )
    parser.add_argument("--prediction", required=True, help="Path to prediction.jsonl")
    parser.add_argument(
        "--generations",
        help="Path to generations.jsonl. Defaults to the file next to prediction.jsonl.",
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset parquet with index and train_token. Defaults to config.resolved.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_positive_predictions(
        prediction_path=Path(args.prediction),
        generations_path=Path(args.generations) if args.generations else None,
        dataset_path=Path(args.dataset) if args.dataset else None,
    )


if __name__ == "__main__":
    main()
