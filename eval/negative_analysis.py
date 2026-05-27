#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable


PRIMARY_FINISH_REASONS = ("length", "stop")
OUTPUT_FILENAME = "_negative_analysis.txt"


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


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _format_counter(title: str, counter: Counter[str], indent: str = "") -> list[str]:
    lines = [f"{indent}{title}:"]
    if not counter:
        lines.append(f"{indent}  <none>: 0")
        return lines

    for key, count in counter.most_common():
        lines.append(f"{indent}  {key}: {count}")
    return lines


def _counter_dict(counter: Counter[str]) -> Dict[str, int]:
    return dict(counter.most_common())


def _output_tokens(rows: list[Dict[str, Any]]) -> list[float]:
    return [
        token
        for token in (_numeric_value(row.get("output_token_length")) for row in rows)
        if token is not None
    ]


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


def _format_overall_output_token_stats(rows: list[Dict[str, Any]]) -> list[str]:
    tokens = _output_tokens(rows)
    lines = ["Overall statistic:"]
    if not tokens:
        lines.append("  count: 0")
        return lines

    q1, q2, q3 = _quantiles(tokens)
    lines.extend(
        [
            f"  count: {len(tokens)}",
            f"  avg output_token: {_format_float(mean(tokens))}",
            f"  q1: {_format_float(q1)}",
            f"  q2: {_format_float(q2)}",
            f"  q3: {_format_float(q3)}",
        ]
    )
    return lines


def _format_output_token_summary(rows: list[Dict[str, Any]], indent: str = "") -> list[str]:
    tokens = _output_tokens(rows)
    if not tokens:
        return [f"{indent}avg output_token: <none>"]

    q1, q2, q3 = _quantiles(tokens)
    return [
        f"{indent}avg output_token: {_format_float(mean(tokens))}",
        f"{indent}q1: {_format_float(q1)}",
        f"{indent}q2: {_format_float(q2)}",
        f"{indent}q3: {_format_float(q3)}",
    ]


def _build_analysis(prediction_path: Path, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    negative_rows = [row for row in rows if _is_incorrect(row.get("is_correct"))]
    finish_counter = Counter(_value_label(row.get("finish_reason")) for row in negative_rows)
    rows_by_finish = {
        finish_reason: [
            row
            for row in negative_rows
            if _value_label(row.get("finish_reason")) == finish_reason
        ]
        for finish_reason in sorted(finish_counter)
    }

    finish_order = list(PRIMARY_FINISH_REASONS)
    finish_order.extend(
        finish_reason
        for finish_reason in sorted(rows_by_finish)
        if finish_reason not in PRIMARY_FINISH_REASONS
    )

    return {
        "prediction": str(prediction_path),
        "total_rows": len(rows),
        "incorrect_rows": len(negative_rows),
        "incorrect_by_finish_reason": _counter_dict(finish_counter),
        "finish_order": finish_order,
        "rows_by_finish": rows_by_finish,
        "negative_rows": negative_rows,
    }


def _format_index_list(rows: list[Dict[str, Any]], indent: str) -> list[str]:
    indexes = [_value_label(row.get("index")) for row in rows]
    if not indexes:
        return []
    return [f"{indent}indexes: {', '.join(indexes)}"]


def _format_can_not_extract(rows: list[Dict[str, Any]], indent: str) -> list[str]:
    lines = [f"{indent}- match reason = can_not_extract: {len(rows)}"]
    lines.extend(
        _format_counter(
            "think_type",
            Counter(_value_label(row.get("think_type")) for row in rows),
            indent=f"{indent}  ",
        )
    )
    return lines


def _format_no_match(rows: list[Dict[str, Any]], indent: str) -> list[str]:
    lines = [f"{indent}- match reason = no_match: {len(rows)}"]
    rows_by_box_source: dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_box_source.setdefault(_value_label(row.get("last_box_source")), []).append(row)

    for box_source in sorted(rows_by_box_source):
        box_rows = rows_by_box_source[box_source]
        lines.append(f"{indent}  - last_box_source = {box_source}: {len(box_rows)}")
        lines.extend(
            _format_counter(
                "think_type",
                Counter(_value_label(row.get("think_type")) for row in box_rows),
                indent=f"{indent}    ",
            )
        )
        if box_source in {"none", "<missing>"}:
            lines.extend(_format_index_list(box_rows, indent=f"{indent}    "))

    return lines


def _format_other_reason(reason: str, rows: list[Dict[str, Any]], indent: str) -> list[str]:
    lines = [f"{indent}- match reason = {reason}: {len(rows)}"]
    lines.extend(
        _format_counter(
            "think_type",
            Counter(_value_label(row.get("think_type")) for row in rows),
            indent=f"{indent}  ",
        )
    )
    lines.extend(
        _format_counter(
            "last_box_source",
            Counter(_value_label(row.get("last_box_source")) for row in rows),
            indent=f"{indent}  ",
        )
    )
    return lines


def _format_analysis(analysis: Dict[str, Any]) -> str:
    lines = [
        f"Prediction: {analysis['prediction']}",
        f"Total rows: {analysis['total_rows']}",
        f"Incorrect rows: {analysis['incorrect_rows']}",
    ]
    lines.extend(
        _format_counter(
            "Incorrect by finish_reason",
            Counter(analysis["incorrect_by_finish_reason"]),
        )
    )
    lines.append("")
    lines.extend(_format_overall_output_token_stats(analysis["negative_rows"]))

    for finish_reason in analysis["finish_order"]:
        finish_rows = analysis["rows_by_finish"].get(finish_reason, [])
        rows_by_reason: dict[str, list[Dict[str, Any]]] = {}
        for row in finish_rows:
            rows_by_reason.setdefault(_value_label(row.get("reason")), []).append(row)

        lines.append("")
        lines.append(f"finish_reason={finish_reason}: {len(finish_rows)}")
        lines.extend(_format_output_token_summary(finish_rows, indent="  "))

        if "can_not_extract" in rows_by_reason:
            lines.extend(_format_can_not_extract(rows_by_reason["can_not_extract"], indent="  "))

        if "no_match" in rows_by_reason:
            lines.extend(_format_no_match(rows_by_reason["no_match"], indent="  "))

        for reason in sorted(set(rows_by_reason) - {"can_not_extract", "no_match"}):
            lines.extend(_format_other_reason(reason, rows_by_reason[reason], indent="  "))

    return "\n".join(lines) + "\n"


def analyze_negative_predictions(prediction_path: Path) -> None:
    rows = list(_iter_jsonl(prediction_path))
    analysis = _build_analysis(prediction_path, rows)
    output_path = prediction_path.with_name(OUTPUT_FILENAME)
    analysis_text = _format_analysis(analysis)
    output_path.write_text(analysis_text, encoding="utf-8")
    print(analysis_text, end="")
    print()
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze incorrect rows in prediction.jsonl by finish reason."
    )
    parser.add_argument("--prediction", required=True, help="Path to prediction.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_negative_predictions(Path(args.prediction))


if __name__ == "__main__":
    main()
