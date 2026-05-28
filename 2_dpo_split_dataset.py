#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_DATA_PATH = "math_dapo_processed.parquet"
DEFAULT_OUTPUT_ROOT = "overeasy_candidates"
DEFAULT_RUN_ID = "overeasy_candidate"
DEFAULT_SYSTEM_PROMPT = (
    r"You are a careful math assistant. Think in <think>...</think>. "
    r"After </think>, give a concise solution and end with the final answer inside \boxed{}."
)
SEED_MIN = 0
SEED_MAX = 2**31 - 1

FIXED_SAMPLING_CONFIG = {
    "greedy": {
        "temperature": 0.0,
        "top_p": 1.0,
        "n_samples": 1,
    },
    "low_temp": {
        "temperature": 0.2,
        "top_p": 0.95,
        "n_samples": 3,
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "run_id": DEFAULT_RUN_ID,
    "output_root": DEFAULT_OUTPUT_ROOT,
    "dataset": {
        "data_path": DEFAULT_DATA_PATH,
    },
    "inference": {
        "enable_chunked_prefill": True,
        "gpu_memory_utilization": 0.9,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "enable_thinking": True,
        "trust_remote_code": True,
        "enforce_eager": False,
        "attention_backend": None,
        "max_model_len": None,
        "max_new_tokens": 2048,
        "batch_size": 512,
    },
    "models": {},
    "sampling": FIXED_SAMPLING_CONFIG,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
}


def load_records(dataset_config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_path = dataset_config["data_path"]
    suffix = Path(dataset_path).suffix.lower()

    if suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {dataset_path}")
        return data

    if suffix == ".jsonl":
        records = []
        with open(dataset_path, "r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {dataset_path}:{line_number}: {exc}") from exc
        return records

    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=dataset_path)["train"]
        return [dict(item) for item in dataset]

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def get_question(record: dict[str, Any]) -> str:
    if "question" not in record or record["question"] is None:
        raise ValueError(f"Record is missing required question field: {record}")
    return str(record["question"])


def get_index(record: dict[str, Any]) -> Any:
    if "index" not in record or record["index"] is None:
        raise ValueError(f"Record is missing required index field: {record}")
    return record["index"]


def get_label(record: dict[str, Any]) -> Any:
    if "label" not in record:
        raise ValueError(f"Record is missing required label field: {record}")
    return record["label"]


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    with open(config_path, "r", encoding="utf-8") as file:
        overrides = json.load(file)
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    return deep_update(config, overrides)


def normalize_optional_revision(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    inference = config.setdefault("inference", {})
    dataset = config.setdefault("dataset", {})
    models = config.setdefault("models", {})

    if args.run_id:
        config["run_id"] = args.run_id
    if args.output_root:
        config["output_root"] = args.output_root
    if args.data_path:
        dataset["data_path"] = args.data_path
    if args.student:
        models["student"] = args.student
    if args.revision is not None:
        models["revision"] = normalize_optional_revision(args.revision)
    if args.max_tokens is not None:
        inference["max_new_tokens"] = args.max_tokens
    if args.max_model_len is not None:
        inference["max_model_len"] = args.max_model_len
    if args.limit is not None:
        dataset["limit"] = args.limit
    if args.batch_size is not None:
        inference["batch_size"] = args.batch_size

    return config


def generate_random_seeds(num_samples: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    rng = secrets.SystemRandom()
    seeds: set[int] = set()
    while len(seeds) < num_samples:
        seeds.add(rng.randint(SEED_MIN, SEED_MAX))
    return list(seeds)


def prepare_sampling_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    sampling_config = copy.deepcopy(FIXED_SAMPLING_CONFIG)
    config["sampling"] = sampling_config

    runs = [
        {
            "sample_type": "greedy",
            "sample_index": 0,
            "temperature": sampling_config["greedy"]["temperature"],
            "top_p": sampling_config["greedy"]["top_p"],
            "seed": None,
        }
    ]

    low_temp = sampling_config["low_temp"]
    low_temp_seeds = generate_random_seeds(int(low_temp["n_samples"]))
    low_temp["seeds"] = low_temp_seeds

    for offset, seed in enumerate(low_temp_seeds, start=1):
        runs.append(
            {
                "sample_type": "low_temp",
                "sample_index": offset,
                "temperature": low_temp["temperature"],
                "top_p": low_temp["top_p"],
                "seed": seed,
            }
        )

    return runs


def load_tokenizer_and_vllm(config: dict[str, Any]):
    model_path = config["models"].get("student") or config["models"].get("model")
    model_revision = normalize_optional_revision(config["models"].get("revision"))
    config["models"]["revision"] = model_revision
    if not model_path:
        raise ValueError("Config must define models.student/models.model or pass --model")

    logging.info("Loading ckpt and tokenizer: %s", model_path)
    if model_revision:
        logging.info("Using model/tokenizer revision: %s", model_revision)

    tokenizer_kwargs = {"trust_remote_code": True}
    if model_revision:
        tokenizer_kwargs["revision"] = model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "left"

    if tokenizer.eos_token is None:
        raise ValueError("No available eos_token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("tokenizer's eos_token: %s, pad_token: %s", tokenizer.eos_token, tokenizer.pad_token)
    logging.info("tokenizer's eos_token_id: %s, pad_token_id: %s", tokenizer.eos_token_id, tokenizer.pad_token_id)

    inference = config["inference"]
    attention_backend = inference.get("attention_backend")
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
        logging.info("Using vLLM attention backend: %s", attention_backend)

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_chunked_prefill=inference["enable_chunked_prefill"],
        gpu_memory_utilization=inference["gpu_memory_utilization"],
        trust_remote_code=inference["trust_remote_code"],
        dtype=torch.bfloat16,
        enforce_eager=inference["enforce_eager"],
        max_model_len=inference.get("max_model_len"),
    )
    if model_revision:
        llm_kwargs["revision"] = model_revision

    llm = LLM(**llm_kwargs)
    logging.info("vLLM model loaded successfully")
    return tokenizer, llm


def render_inputs(records: list[dict[str, Any]], config: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    rendered = []
    system_prompt = config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    enable_thinking = config["inference"].get("enable_thinking", True)

    for record in records:
        question = get_question(record)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        rendered.append(
            {
                "index": get_index(record),
                "question": question,
                "label": get_label(record),
                "input_text": input_text,
            }
        )

    return rendered


def prepare_run_dir(config: dict[str, Any]) -> Path:
    run_id = config.get("run_id")
    if not run_id:
        raise ValueError("Config must define run_id or pass --run-id")

    run_dir = Path(config.get("output_root", DEFAULT_OUTPUT_ROOT)) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(config: dict[str, Any], run_dir: Path) -> Path:
    resolved_path = run_dir / "config.resolved.json"
    with open(resolved_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return resolved_path


def build_sampling_params(config: dict[str, Any], sampling_run: dict[str, Any]) -> SamplingParams:
    inference = config["inference"]
    return SamplingParams(
        n=1,
        top_p=sampling_run["top_p"],
        min_p=inference.get("min_p", 0.0),
        temperature=sampling_run["temperature"],
        presence_penalty=inference.get("presence_penalty", 0.0),
        seed=sampling_run["seed"],
        skip_special_tokens=False,
        ignore_eos=False,
        max_tokens=int(inference["max_new_tokens"]),
        stop=["<|im_end|>"],
    )


def get_limited_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = load_records(config["dataset"])
    limit = config["dataset"].get("limit")
    if limit is not None:
        records = records[: int(limit)]
    return records


def build_generation_row(
    config: dict[str, Any],
    item: dict[str, Any],
    first_output: Any,
    sampling_run: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": config["run_id"],
        "index": item["index"],
        "question": item["question"],
        "label": item["label"],
        "sample_type": sampling_run["sample_type"],
        "sample_index": sampling_run["sample_index"],
        "temperature": sampling_run["temperature"],
        "top_p": sampling_run["top_p"],
        "seed": sampling_run["seed"],
        "model_output": first_output.text,
        "input_text": item["input_text"],
        "finish_reason": getattr(first_output, "finish_reason", None),
        "output_token_length": len(getattr(first_output, "token_ids", []) or []),
    }


def generate(config: dict[str, Any]) -> None:
    sampling_runs = prepare_sampling_runs(config)
    records = get_limited_records(config)
    tokenizer, llm = load_tokenizer_and_vllm(config)
    rendered = render_inputs(records, config, tokenizer)

    run_dir = prepare_run_dir(config)
    resolved_config_path = write_resolved_config(config, run_dir)

    generations_path = run_dir / "generations.jsonl"
    batch_size = int(config["inference"].get("batch_size", 128))
    max_new_tokens = int(config["inference"]["max_new_tokens"])

    logging.info(
        "Starting overeasy candidate generation with %d records, %d candidates per record, max_new_tokens=%d",
        len(rendered),
        len(sampling_runs),
        max_new_tokens,
    )
    logging.info("Resolved config written to %s", resolved_config_path)
    logging.info("Low-temp seeds: %s", config["sampling"]["low_temp"]["seeds"])

    with open(generations_path, "w", encoding="utf-8") as generations_file:
        for start in tqdm(range(0, len(rendered), batch_size), desc="Generating overeasy candidates"):
            batch = rendered[start:start + batch_size]
            input_texts = [item["input_text"] for item in batch]

            for sampling_run in sampling_runs:
                sampling_params = build_sampling_params(config, sampling_run)
                outputs = llm.generate(input_texts, sampling_params)

                for item, output in zip(batch, outputs):
                    first_output = output.outputs[0]
                    row = build_generation_row(config, item, first_output, sampling_run)
                    generations_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Generations written to %s", generations_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1 greedy + 3 random low-temp candidates per math problem.")
    parser.add_argument("--config", type=Path, default=None, help="optional JSON config path")
    parser.add_argument("--run-id", type=str, help="override run_id")
    parser.add_argument("--output-root", type=str, help="override output root directory")
    parser.add_argument("--data-path", type=str, help=f"override dataset path, default {DEFAULT_DATA_PATH}")
    parser.add_argument("--model", "--student", dest="student", type=str, default=None, help="student/model path")
    parser.add_argument("--revision", type=str, default=None, help="model/tokenizer revision; use null/none to disable")
    parser.add_argument("--max-tokens", type=int, help="override inference.max_new_tokens")
    parser.add_argument("--max-model-len", type=int, help="override inference.max_model_len")
    parser.add_argument("--limit", type=int, help="limit number of input records")
    parser.add_argument("--batch-size", type=int, help="override inference.batch_size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    generate(config)


if __name__ == "__main__":
    main()
