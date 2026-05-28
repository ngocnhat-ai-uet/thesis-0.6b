#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import secrets
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EASY_FILENAME = "easy.jsonl"
HARD_FILENAME = "hard.jsonl"
MEDIUM_FILENAME = "medium.jsonl"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def value_label(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    return text


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=lambda item: str(item))}


def difficulty_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return counter_dict(Counter(value_label(row.get("difficulty")) for row in rows))


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truthy_one(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return False


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


def classify_prediction_group(rows: list[dict[str, Any]]) -> str:
    is_easy = all(
        truthy_one(row.get("is_correct")) and optional_text(row.get("finish_reason")) == "stop"
        for row in rows
    )
    if is_easy:
        return "easy"

    is_hard = all(str(row.get("reason")) == "can_not_extract" for row in rows)
    if is_hard:
        return "hard"

    return "medium"


def validate_split_inputs(
    dataset_rows: list[dict[str, Any]],
    predictions_by_index: dict[str, list[dict[str, Any]]],
    dataset_path: Path,
    prediction_path: Path,
) -> None:
    dataset_indexes = {str(row["index"]) for row in dataset_rows}
    prediction_indexes = set(predictions_by_index)
    missing_predictions = sorted(dataset_indexes - prediction_indexes)
    extra_predictions = sorted(prediction_indexes - dataset_indexes)
    if missing_predictions:
        raise ValueError(
            f"{prediction_path} is missing predictions for {len(missing_predictions)} "
            f"dataset rows from {dataset_path}. Examples: {missing_predictions[:5]}"
        )
    if extra_predictions:
        raise ValueError(
            f"{prediction_path} contains {len(extra_predictions)} indexes not present in "
            f"{dataset_path}. Examples: {extra_predictions[:5]}"
        )


def split_dataset(dataset_path: Path, prediction_path: Path, output_dir: Path) -> None:
    dataset_rows = load_unique_dataset(dataset_path)
    predictions_by_index = load_predictions_by_index(prediction_path)
    validate_split_inputs(dataset_rows, predictions_by_index, dataset_path, prediction_path)

    split_rows: dict[str, list[dict[str, Any]]] = {
        "easy": [],
        "hard": [],
        "medium": [],
    }
    prediction_group_sizes: Counter[int] = Counter()

    for row in dataset_rows:
        index = str(row["index"])
        prediction_rows = predictions_by_index[index]
        prediction_group_sizes[len(prediction_rows)] += 1
        split_rows[classify_prediction_group(prediction_rows)].append(row)

    output_paths = {
        "easy": output_dir / EASY_FILENAME,
        "hard": output_dir / HARD_FILENAME,
        "medium": output_dir / MEDIUM_FILENAME,
    }
    output_counts = {
        split_name: write_jsonl(output_paths[split_name], rows)
        for split_name, rows in split_rows.items()
    }

    manifest = {
        "dataset": str(dataset_path),
        "prediction": str(prediction_path),
        "output_dir": str(output_dir),
        "dataset_rows": len(dataset_rows),
        "prediction_rows": sum(len(rows) for rows in predictions_by_index.values()),
        "prediction_indexes": len(predictions_by_index),
        "prediction_group_sizes": {
            str(size): count for size, count in sorted(prediction_group_sizes.items())
        },
        "split_rules": {
            "easy": "all prediction rows for index have is_correct=1 and finish_reason=stop",
            "hard": "all prediction rows for index have reason=can_not_extract",
            "medium": "all remaining indexes not in easy/hard",
        },
        "outputs": {
            split_name: {
                "path": str(output_paths[split_name]),
                "rows": output_counts[split_name],
                "difficulty": difficulty_distribution(split_rows[split_name]),
            }
            for split_name in output_paths
        },
    }
    write_json(output_dir / SPLIT_MANIFEST_FILENAME, manifest)

    print(f"Wrote split dataset to {output_dir}")
    for split_name in ("easy", "medium", "hard"):
        print(f"{split_name}: {output_counts[split_name]}")


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("run_id")),
        str(row.get("index")),
        str(row.get("sample_index")),
        str(row.get("seed")),
    )


def has_extracted_answer(row: dict[str, Any]) -> bool:
    value = row.get("extracted_answer")
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def vr_score(row: dict[str, Any]) -> int:
    if has_extracted_answer(row) and str(row.get("reason")) != "can_not_extract":
        return 1 if truthy_one(row.get("is_correct")) else 0
    return -1


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


def pair_length(pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
    prediction, generation = pair
    token_length = numeric_value(prediction.get("output_token_length"))
    if token_length is not None:
        return token_length
    output = model_output_text(generation)
    return float(len(output) if output is not None else 0)


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
    rng = secrets.SystemRandom()

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
            grouped_pairs[vr_score(pair[0])].append(pair)

        positive_pairs = grouped_pairs[1]
        zero_pairs = grouped_pairs[0]
        negative_pairs = grouped_pairs[-1]

        if not positive_pairs and zero_pairs and not negative_pairs:
            skip_reasons["all_zero_samples"] += 1
            continue

        if positive_pairs:
            chosen_pair = longest_pair(positive_pairs)
            chosen_vr_score = 1
        elif zero_pairs:
            chosen_pair = longest_pair(zero_pairs)
            chosen_vr_score = 0
        else:
            skip_reasons["missing_chosen"] += 1
            continue

        rejected_pair = None
        rejected_vr_score = None
        if negative_pairs:
            rejected_pair = rng.choice(negative_pairs)
            rejected_vr_score = -1
        elif chosen_vr_score == 1 and zero_pairs:
            rejected_pair = rng.choice(zero_pairs)
            rejected_vr_score = 0

        if rejected_pair is None or rejected_vr_score is None:
            skip_reasons["missing_rejected"] += 1
            continue

        chosen_prediction, chosen_generation = chosen_pair
        rejected_prediction, rejected_generation = rejected_pair
        rejected_finish_reason = optional_text(rejected_prediction.get("finish_reason"))
        length_reject = rejected_finish_reason == "length"
        chosen_answer = optional_text(chosen_prediction.get("extracted_answer"))
        dpo_rows.append(
            {
                "index": record["index"],
                "label": record.get("label", chosen_prediction.get("label")),
                "correct_answer": (
                    chosen_answer
                    if chosen_vr_score == 1
                    else optional_text(record.get("label") or chosen_prediction.get("label"))
                ),
                "wrong_answer": optional_text(rejected_prediction.get("extracted_answer")),
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

    print(f"Wrote {len(dpo_rows)} DPO rows to {output_path}")
    if skip_reasons:
        print(f"Skipped: {dict(sorted(skip_reasons.items()))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split DPO inference datasets and build DPO chosen/rejected pairs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="split raw dataset by prediction outcomes")
    split_parser.add_argument("--dataset", type=Path, required=True, help="input raw dataset JSONL")
    split_parser.add_argument("--prediction", type=Path, required=True, help="prediction.jsonl")
    split_parser.add_argument("--output-dir", type=Path, required=True, help="output split directory")

    dpo_parser = subparsers.add_parser("build-dpo", help="build dpo.py-compatible JSON dataset")
    dpo_parser.add_argument("--dataset", type=Path, required=True, help="input dataset JSONL")
    dpo_parser.add_argument("--prediction", type=Path, required=True, help="prediction.jsonl")
    dpo_parser.add_argument("--generations", type=Path, required=True, help="generations.jsonl")
    dpo_parser.add_argument("--output", type=Path, required=True, help="output DPO JSON")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "split":
        split_dataset(args.dataset, args.prediction, args.output_dir)
        return
    if args.command == "build-dpo":
        build_dpo_dataset(args.dataset, args.prediction, args.generations, args.output)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
