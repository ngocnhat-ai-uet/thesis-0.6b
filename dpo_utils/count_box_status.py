#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


THINK_CLOSE = "</think>"


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import matcher  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc


def valid_boxed_answers(text: Any) -> list[matcher.BoxedAnswer]:
    return matcher._valid_boxed_answers(text)  # Reuse the balanced-brace parser in matcher.py.


def has_nonempty_box(text: Any) -> bool:
    return any(answer.content.strip() for answer in valid_boxed_answers(text))


def has_valid_policy_box(text: Any, finish_reason: Any) -> bool:
    full_text = matcher.to_text(text)
    reason = matcher.to_text(finish_reason).strip().lower()

    if reason == "length":
        close_idx = full_text.find(THINK_CLOSE)
        if close_idx == -1:
            return False
        answers = valid_boxed_answers(full_text[:close_idx])
    else:
        answers = valid_boxed_answers(full_text)

    nonempty_answers = [answer for answer in answers if answer.content.strip()]
    if not nonempty_answers:
        return False

    return nonempty_answers[-1] == answers[-1]


def count_box_status(input_path: Path, output_field: str, finish_reason_field: str) -> Counter[str]:
    counts: Counter[str] = Counter()

    for row in iter_jsonl(input_path):
        counts["total"] += 1
        model_output = row.get(output_field, "")
        finish_reason = row.get(finish_reason_field, "")

        if not has_nonempty_box(model_output):
            counts["no_box"] += 1
            continue

        counts["has_box"] += 1
        if has_valid_policy_box(model_output, finish_reason):
            counts["valid_box"] += 1
        else:
            counts["invalid_box"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count rows with no boxed answer, non-empty boxed answer, and policy-valid boxed answer."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).with_name("generations.jsonl")),
        help="Path to generations.jsonl",
    )
    parser.add_argument("--output-field", default="model_output", help="Field containing model output text")
    parser.add_argument("--finish-reason-field", default="finish_reason", help="Field containing finish reason")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    counts = count_box_status(input_path, args.output_field, args.finish_reason_field)

    print(f"Input: {input_path}")
    print(f"Total: {counts['total']}")
    print(f"Khong box: {counts['no_box']}")
    print(f"Co box: {counts['has_box']}")
    print(f"Co box hop le: {counts['valid_box']}")
    print(f"Co box khong hop le: {counts['invalid_box']}")


if __name__ == "__main__":
    main()
