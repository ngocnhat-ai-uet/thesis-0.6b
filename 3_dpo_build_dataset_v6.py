#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent
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
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield row


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(yaml_lines(value)))
        file.write("\n")


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_required_index(row: dict[str, Any], path: Path, row_number: int) -> str:
    if "index" not in row or row["index"] is None:
        raise ValueError(f"Missing index at {path}:{row_number}")
    return str(row["index"])


def load_unique_dataset(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        index = get_required_index(row, path, row_number)
        if index in seen:
            duplicates.append(index)
        seen.add(index)
    if duplicates:
        raise ValueError(f"{path} contains duplicate index values. Examples: {duplicates[:5]}")
    return rows


def load_predictions_by_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        index = get_required_index(row, path, row_number)
        groups[index].append(row)
    if not groups:
        raise ValueError(f"{path} contains no prediction rows")
    return dict(groups)


def generation_key(row: dict[str, Any], path: Path, row_number: int) -> tuple[str, str, str, str]:
    missing = [
        field
        for field in ("run_id", "index", "sample_index", "seed")
        if field not in row or row[field] is None
    ]
    if missing:
        raise ValueError(f"Missing {missing} at {path}:{row_number}")
    return (
        str(row["run_id"]),
        str(row["index"]),
        str(row["sample_index"]),
        str(row["seed"]),
    )


def load_generations_by_key(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    generations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str, str, str]] = []
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        key = generation_key(row, path, row_number)
        if key in generations:
            duplicates.append(key)
        generations[key] = row
    if duplicates:
        raise ValueError(f"{path} contains duplicate generation keys. Examples: {duplicates[:5]}")
    if not generations:
        raise ValueError(f"{path} contains no generation rows")
    return generations


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("run_id")),
        str(row.get("index")),
        str(row.get("sample_index")),
        str(row.get("seed")),
    )


def model_output_text(generation: dict[str, Any]) -> str | None:
    value = generation.get("model_output")
    if value is None:
        return None
    text = str(value)
    if not text.strip():
        return None
    return text


def candidates_with_generation(
    candidates: list[dict[str, Any]],
    generations_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    valid_pairs = []
    for prediction in candidates:
        generation = generations_by_key.get(prediction_key(prediction))
        if generation is None:
            continue
        if model_output_text(generation) is not None:
            valid_pairs.append((prediction, generation))
    return valid_pairs


def pair_token_length(pair: tuple[dict[str, Any], dict[str, Any]]) -> int:
    prediction, generation = pair
    token_length = numeric_value(prediction.get("output_token_length"))
    if token_length is not None:
        return int(token_length)
    token_length = numeric_value(generation.get("output_token_length"))
    if token_length is not None:
        return int(token_length)
    output = model_output_text(generation)
    return len(output) if output is not None else 0


def pair_length(pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
    return float(pair_token_length(pair))


def longest_pair(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return max(pairs, key=pair_length)


def get_prompt(record: dict[str, Any]) -> str:
    for field in ("question", "prompt", "instruction"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def pair_finish_reason(pair: tuple[dict[str, Any], dict[str, Any]]) -> str | None:
    prediction, generation = pair
    return optional_text(prediction.get("finish_reason") or generation.get("finish_reason"))


def pair_policy_box(pair: tuple[dict[str, Any], dict[str, Any]]) -> Any:
    _, generation = pair
    return matcher.find_policy_boxed_answer(
        generation.get("model_output"),
        pair_finish_reason(pair),
    )


def pair_vr_score(
    pair: tuple[dict[str, Any], dict[str, Any]],
    label: Any,
    question: Any,
) -> int | None:
    _, generation = pair
    output = generation.get("model_output")
    finish_reason = pair_finish_reason(pair)
    finish_reason_text = matcher.to_text(finish_reason).strip().lower()
    last_box = matcher.find_last_boxed_answer(output)
    if finish_reason_text == "length" and "</think>" not in matcher.to_text(output):
        return None if last_box.found else -1

    answer = matcher.find_policy_boxed_answer(output, finish_reason)
    if not answer.found:
        return -1 if not last_box.found else None
    if not answer.content.strip():
        return 0

    result = matcher.match_answer(label, output, question, finish_reason=finish_reason)
    return 1 if result.matched else 0


def closest_length_pair(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    target_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return min(
        pairs,
        key=lambda pair: (
            abs(pair_token_length(pair) - target_length),
            pair_token_length(pair),
        ),
    )


def percentile(sorted_values: list[int], percent: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def token_stats(values: list[int]) -> dict[str, Any]:
    sorted_values = sorted(values)
    count = len(sorted_values)
    total = sum(sorted_values)
    return {
        "count": count,
        "mean": round(total / count, 2) if count else None,
        "median": round(percentile(sorted_values, 50), 2) if count else None,
    }


def stats_path_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_stats.yaml")


def print_token_stats(name: str, stats: dict[str, Any]) -> None:
    print(
        f"{name} tokens: count={stats['count']}, "
        f"mean={stats['mean']}, median={stats['median']}"
    )


def build_dpo_dataset(
    dataset_path: Path,
    prediction_path: Path,
    generations_path: Path,
    output_path: Path,
) -> None:
    dataset_rows = load_unique_dataset(dataset_path)
    predictions_by_index = load_predictions_by_index(prediction_path)
    generations_by_key = load_generations_by_key(generations_path)

    dpo_rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    chosen_score_counts: Counter[int] = Counter()
    rejected_score_counts: Counter[int] = Counter()
    pair_score_counts: Counter[tuple[int, int]] = Counter()
    chosen_token_lengths: list[int] = []
    rejected_token_lengths: list[int] = []
    both_under_2048_count = 0
    rng = random.Random(9)

    for record in dataset_rows:
        index = str(record["index"])
        prompt = get_prompt(record)
        prediction_rows = predictions_by_index.get(index)
        if not prediction_rows:
            skip_reasons["missing_prediction"] += 1
            continue
        if not prompt:
            skip_reasons["missing_prompt"] += 1
            continue

        valid_pairs = candidates_with_generation(prediction_rows, generations_by_key)
        if not valid_pairs:
            skip_reasons["missing_generation"] += 1
            continue

        grouped_pairs: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {
            1: [],
            0: [],
            -1: [],
        }
        for pair in valid_pairs:
            score = pair_vr_score(pair, record.get("label", pair[0].get("label")), prompt)
            if score is None:
                skip_reasons["invalid_truncated_or_policy_box"] += 1
                continue
            grouped_pairs[score].append(pair)

        positive_pairs = grouped_pairs[1]
        zero_pairs = grouped_pairs[0]

        if positive_pairs:
            chosen_pair = rng.choice(positive_pairs)
            chosen_vr_score = 1
        else:
            skip_reasons["missing_chosen"] += 1
            continue

        rejected_pair = None
        rejected_vr_score = None
        if chosen_vr_score == 1:
            if zero_pairs:
                rejected_pair = closest_length_pair(zero_pairs, pair_token_length(chosen_pair))
                rejected_vr_score = 0

        if rejected_pair is None or rejected_vr_score is None:
            skip_reasons["missing_rejected"] += 1
            continue

        chosen_prediction, chosen_generation = chosen_pair
        rejected_prediction, rejected_generation = rejected_pair
        chosen_token_length = pair_token_length(chosen_pair)
        rejected_token_length = pair_token_length(rejected_pair)
        chosen_score_counts[chosen_vr_score] += 1
        rejected_score_counts[rejected_vr_score] += 1
        pair_score_counts[(chosen_vr_score, rejected_vr_score)] += 1
        chosen_token_lengths.append(chosen_token_length)
        rejected_token_lengths.append(rejected_token_length)
        if chosen_token_length < 2048 and rejected_token_length < 2048:
            both_under_2048_count += 1
        rejected_finish_reason = pair_finish_reason(rejected_pair)
        length_reject = matcher.to_text(rejected_finish_reason).strip().lower() == "length"
        chosen_answer = optional_text(pair_policy_box(chosen_pair).content)
        rejected_answer = optional_text(pair_policy_box(rejected_pair).content)
        dpo_rows.append(
            {
                "index": record["index"],
                "label": record.get("label", chosen_prediction.get("label")),
                "correct_answer": (
                    chosen_answer
                    if chosen_vr_score == 1
                    else optional_text(record.get("label") or chosen_prediction.get("label"))
                ),
                "wrong_answer": rejected_answer,
                "length_reject": length_reject,
                "prompt": prompt,
                "chosen": str(chosen_generation["model_output"]),
                "rejected": str(rejected_generation["model_output"]),
                "chosen_vr_score": chosen_vr_score,
                "rejected_vr_score": rejected_vr_score,
                "chosen_sample_index": chosen_prediction.get("sample_index"),
                "rejected_sample_index": rejected_prediction.get("sample_index"),
                "chosen_seed": chosen_prediction.get("seed"),
                "rejected_seed": rejected_prediction.get("seed"),
            }
        )

    write_json(output_path, dpo_rows)

    chosen_stats = token_stats(chosen_token_lengths)
    rejected_stats = token_stats(rejected_token_lengths)
    stats_path = stats_path_for_output(output_path)
    stats = {
        "dataset": str(dataset_path),
        "prediction": str(prediction_path),
        "generations": str(generations_path),
        "output": str(output_path),
        "rows": len(dpo_rows),
        "skipped": dict(sorted(skip_reasons.items())),
        "chosen_score_counts": {
            str(key): chosen_score_counts[key] for key in sorted(chosen_score_counts)
        },
        "rejected_score_counts": {
            str(key): rejected_score_counts[key] for key in sorted(rejected_score_counts)
        },
        "pair_score_counts": {
            f"{chosen_score},{rejected_score}": pair_score_counts[(chosen_score, rejected_score)]
            for chosen_score, rejected_score in sorted(pair_score_counts)
        },
        "chosen_tokens": chosen_stats,
        "rejected_tokens": rejected_stats,
        "pairs_both_under_2048": both_under_2048_count,
    }
    write_yaml(stats_path, stats)

    print(f"Wrote {len(dpo_rows)} DPO rows to {output_path}")
    print(f"Wrote stats to {stats_path}")
    print_token_stats("Chosen", chosen_stats)
    print_token_stats("Rejected", rejected_stats)
    print(f"Pairs with chosen and rejected tokens < 2048: {both_under_2048_count}")
    if skip_reasons:
        print(f"Skipped: {dict(sorted(skip_reasons.items()))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs.")
    parser.add_argument("--dataset", type=Path, required=True, help="input dataset JSONL")
    parser.add_argument("--prediction", type=Path, required=True, help="prediction.jsonl")
    parser.add_argument("--generations", type=Path, required=True, help="generations.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="output DPO JSON")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dpo_dataset(args.dataset, args.prediction, args.generations, args.output)


if __name__ == "__main__":
    main()
