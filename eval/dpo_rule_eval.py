#!/usr/bin/env python
"""
DPO sample evaluator.

Input rows use `label` as ground truth and `model_output` as raw generation.
Writes `prediction.jsonl` next to the input generations file.

Prediction rows contain:
run_id, index, difficulty, sample_index, seed, temperature, output_token_length,
finish_reason, label, extracted_answer, is_correct, reason, correct_format,
question.

`correct_format` is 1 only when generation stopped normally, has a closed
<think>...</think> block, and the last valid boxed answer is after </think>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matcher as matcher


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

PREDICTION_FILENAME = "prediction.jsonl"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc


def has_closed_think_pair(text: Any) -> bool:
    s = matcher.to_text(text)
    open_idx = s.find(THINK_OPEN)
    if open_idx == -1:
        return False
    close_idx = s.find(THINK_CLOSE, open_idx + len(THINK_OPEN))
    return close_idx != -1


def has_valid_box_after_think(text: Any) -> bool:
    s = matcher.to_text(text)
    close_idx = s.find(THINK_CLOSE)
    if close_idx == -1:
        return False

    solution_text = s[close_idx + len(THINK_CLOSE):]
    marker = r"\boxed{"
    start = solution_text.rfind(marker)
    while start != -1:
        depth = 1
        idx = start + len(marker)
        while idx < len(solution_text):
            char = solution_text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return True
            idx += 1
        start = solution_text.rfind(marker, 0, start)
    return False


def is_correct_format(model_output: Any, finish_reason: Any) -> int:
    if matcher.to_text(finish_reason) != "stop":
        return 0
    if not has_closed_think_pair(model_output):
        return 0
    if not has_valid_box_after_think(model_output):
        return 0
    return 1


def normalized_extracted_answer(result: Any) -> str | None:
    if result.reason == matcher.REASON_CAN_NOT_EXTRACT:
        return None
    extracted_answer = result.extracted_answer
    if extracted_answer is None:
        return None
    if not matcher.to_text(extracted_answer).strip():
        return None
    return extracted_answer


def _build_prediction_row(
    *,
    row: Dict[str, Any],
    row_run_id: str,
    index: Any,
    question: str,
    label: Any,
    result: Any,
    correct_format: int,
) -> Dict[str, Any]:
    return {
        "run_id": row_run_id,
        "index": index,
        "difficulty": row.get("difficulty"),
        "sample_index": row.get("sample_index"),
        "seed": row.get("seed"),
        "temperature": row.get("temperature"),
        "output_token_length": row.get("output_token_length"),
        "finish_reason": row.get("finish_reason"),
        "label": matcher.to_text(label),
        "extracted_answer": normalized_extracted_answer(result),
        "is_correct": 1 if result.matched else 0,
        "reason": result.reason,
        "correct_format": correct_format,
        "question": question,
    }


def evaluate_file(
    input_path: Path,
    label_field: str = "label",
    pred_field: str = "model_output",
    question_field: str = "question",
    run_id_field: str = "run_id",
    index_field: str = "index",
    log_every: int = 100,
    enable_symbolic: bool = False,
) -> None:
    if not enable_symbolic:
        matcher.SYMPY_AVAILABLE = False

    output_dir = input_path.parent
    prediction_path = output_dir / PREDICTION_FILENAME
    total = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    with prediction_path.open("w", encoding="utf-8") as prediction_file:
        for row_index, row in enumerate(_iter_jsonl(input_path)):
            total += 1

            label = row.get(label_field, "")
            model_output = matcher.to_text(row.get(pred_field, ""))
            question = matcher.to_text(row.get(question_field, ""))
            row_run_id = matcher.to_text(row.get(run_id_field, ""))
            index = row.get(index_field, row_index)

            result = matcher.match_answer(label, model_output, question, finish_reason=row.get("finish_reason"))
            prediction_row = _build_prediction_row(
                row=row,
                row_run_id=row_run_id,
                index=index,
                question=question,
                label=label,
                result=result,
                correct_format=is_correct_format(model_output, row.get("finish_reason")),
            )
            prediction_file.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")

            if log_every > 0 and total % log_every == 0:
                print(f"Processed line {total}: index={index}, sample_index={row.get('sample_index')}, seed={row.get('seed')}", flush=True)
    print(f"Input: {input_path}")
    print(f"Prediction: {prediction_path}")
    print(f"Total: {total}")
    print(f"SymPy available: {matcher.SYMPY_AVAILABLE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPO candidate generations.")
    parser.add_argument("--input", required=True, help="Path to input generations.jsonl")
    parser.add_argument("--label-field", default="label", help="Ground-truth field name")
    parser.add_argument("--pred-field", default="model_output", help="Prediction field name")
    parser.add_argument("--question-field", default="question", help="Question text field name")
    parser.add_argument("--run-id-field", default="run_id", help="Run-id field name")
    parser.add_argument("--index-field", default="index", help="Question index field name")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N rows; set 0 to disable")
    parser.add_argument("--enable-symbolic", action="store_true", help="Enable slower SymPy symbolic matching")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_file(
        input_path=Path(args.input),
        label_field=args.label_field,
        pred_field=args.pred_field,
        question_field=args.question_field,
        run_id_field=args.run_id_field,
        index_field=args.index_field,
        log_every=args.log_every,
        enable_symbolic=args.enable_symbolic,
    )


if __name__ == "__main__":
    main()
