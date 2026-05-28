#!/usr/bin/env python

from __future__ import annotations

import argparse
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

DEFAULT_CONFIG_PATH = Path("configs/infer_dpo.json")
DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
DEFAULT_NUM_SAMPLES = 8
SEED_MIN = 0
SEED_MAX = 2**31 - 1
SYSTEM_PROMPT_MODE_NONE = "none"
SYSTEM_PROMPT_MODE_SYSTEM = "system"
SYSTEM_PROMPT_MODE_USER = "user"
VALID_SYSTEM_PROMPT_MODES = {
    SYSTEM_PROMPT_MODE_NONE,
    SYSTEM_PROMPT_MODE_SYSTEM,
    SYSTEM_PROMPT_MODE_USER,
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
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=dataset_path)["train"]
        return [dict(item) for item in dataset]

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def get_question(record: dict[str, Any]) -> str:
    for field in ("question", "instruction", "prompt"):
        if field in record and record[field] is not None:
            return str(record[field])
    raise ValueError(f"Record has no question/instruction/prompt field: {record}")


def get_index(record: dict[str, Any], fallback_index: int) -> Any:
    for field in ("index", "id", "question_idx"):
        if field in record and record[field] is not None:
            return record[field]
    return fallback_index


def get_label(record: dict[str, Any]) -> Any:
    return record.get("answer", record.get("final_answer", record.get("label")))


def build_prompt_text(question: str) -> str:
    return f"{question}"


def build_messages(
    question: str,
    system_prompt: str | None = None,
    system_prompt_mode: str = SYSTEM_PROMPT_MODE_SYSTEM,
) -> list[dict[str, str]]:
    user_text = build_prompt_text(question)
    messages = []

    if system_prompt_mode == SYSTEM_PROMPT_MODE_SYSTEM and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif system_prompt_mode == SYSTEM_PROMPT_MODE_USER and system_prompt:
        user_text = f"{user_text}\n{system_prompt}"

    messages.append({"role": "user", "content": user_text})
    return messages


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in value.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def validate_seeds(seeds: list[int]) -> list[int]:
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    invalid = [seed for seed in seeds if seed < SEED_MIN or seed > SEED_MAX]
    if invalid:
        raise ValueError(f"Seeds must be in [{SEED_MIN}, {SEED_MAX}]. Invalid examples: {invalid[:5]}")
    return seeds


def generate_random_seeds(num_samples: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if num_samples > SEED_MAX - SEED_MIN + 1:
        raise ValueError("num_samples exceeds available unique seed range")

    rng = secrets.SystemRandom()
    seeds: set[int] = set()
    while len(seeds) < num_samples:
        seeds.add(rng.randint(SEED_MIN, SEED_MAX))
    return list(seeds)


def get_seed_list(config: dict[str, Any]) -> list[int]:
    inference = config.setdefault("inference", {})
    if "seeds" in inference and inference["seeds"] is not None:
        seeds = [int(seed) for seed in inference["seeds"]]
        if seeds:
            return validate_seeds(seeds)
    if "seed" in inference and inference["seed"] is not None:
        return validate_seeds([int(inference["seed"])])

    num_samples = int(inference.get("num_samples", DEFAULT_NUM_SAMPLES))
    seeds = generate_random_seeds(num_samples)
    inference["num_samples"] = num_samples
    inference["seeds"] = seeds
    return seeds


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    inference = config.setdefault("inference", {})
    models = config.setdefault("models", {})

    if args.run_id:
        config["run_id"] = args.run_id
    if args.student:
        models["student"] = args.student
    if args.revision:
        models["revision"] = args.revision
    if args.seeds:
        inference["seeds"] = parse_seeds(args.seeds)
    if args.num_samples is not None:
        inference["num_samples"] = args.num_samples
    if args.temperature is not None:
        inference["temperature"] = args.temperature
    if args.max_tokens is not None:
        inference["max_new_tokens"] = args.max_tokens
    if args.max_model_len is not None:
        inference["max_model_len"] = args.max_model_len
    if args.limit is not None:
        config.setdefault("dataset", {})["limit"] = args.limit

    return config


def load_tokenizer_and_vllm(config: dict[str, Any]):
    model_path = config["models"].get("student") or config["models"].get("model")
    model_revision = config["models"].get("revision")
    if not model_path:
        raise ValueError("Config must define models.student or models.model")

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
    dataset_config = config.setdefault("dataset", {})
    system_prompt_mode = dataset_config.get("system_prompt_mode", SYSTEM_PROMPT_MODE_SYSTEM)
    if system_prompt_mode not in VALID_SYSTEM_PROMPT_MODES:
        valid_modes = ", ".join(sorted(VALID_SYSTEM_PROMPT_MODES))
        raise ValueError(
            f"Invalid dataset.system_prompt_mode={system_prompt_mode!r}. "
            f"Valid values: {valid_modes}"
        )
    system_prompt = config.get("system_prompt") or dataset_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    for index, record in enumerate(records):
        question = get_question(record)
        messages = build_messages(question, system_prompt, system_prompt_mode)
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["inference"].get("enable_thinking", True),
        )
        rendered.append(
            {
                "index": get_index(record, index),
                "difficulty": record.get("difficulty"),
                "question": question,
                "label": get_label(record),
                "input_text": full_text,
            }
        )
    return rendered


def prepare_run_dir(config: dict[str, Any]) -> Path:
    run_id = config.get("run_id")
    if not run_id:
        raise ValueError("Config must define run_id or --run-id must be provided")

    run_dir = Path(config.get("output_root", "dpo_experiments")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(config: dict[str, Any], run_dir: Path) -> None:
    resolved_path = run_dir / "config.resolved.json"
    with open(resolved_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_sampling_params(config: dict[str, Any], seed: int) -> SamplingParams:
    inference = config["inference"]
    return SamplingParams(
        n=1,
        top_p=inference.get("top_p", 1.0),
        min_p=inference.get("min_p", 0.0),
        temperature=inference["temperature"],
        presence_penalty=inference.get("presence_penalty", 0.0),
        seed=seed,
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


def format_seed(seed: int) -> str:
    return f"{seed:02d}"


def build_generation_row(
    config: dict[str, Any],
    item: dict[str, Any],
    first_output: Any,
    sample_index: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "run_id": config["run_id"],
        "index": item["index"],
        "difficulty": item["difficulty"],
        "sample_index": sample_index,
        "seed": format_seed(seed),
        "temperature": config["inference"]["temperature"],
        "output_token_length": len(getattr(first_output, "token_ids", []) or []),
        "finish_reason": getattr(first_output, "finish_reason", None),
        "label": item["label"],
        "question": item["question"],
        "model_output": first_output.text,
        "input_text": item["input_text"],
    }


def generate(config: dict[str, Any]) -> None:
    seeds = get_seed_list(config)
    records = get_limited_records(config)
    tokenizer, llm = load_tokenizer_and_vllm(config)
    rendered = render_inputs(records, config, tokenizer)

    run_dir = prepare_run_dir(config)
    write_resolved_config(config, run_dir)

    generations_path = run_dir / "generations.jsonl"
    batch_size = int(config["inference"].get("batch_size", 32))
    max_new_tokens = int(config["inference"]["max_new_tokens"])

    logging.info(
        "Starting DPO candidate generation with %d records, %d seeds, max_new_tokens=%d",
        len(rendered),
        len(seeds),
        max_new_tokens,
    )
    logging.info("Seeds: %s", seeds)

    with open(generations_path, "w", encoding="utf-8") as generations_file:
        for start in tqdm(range(0, len(rendered), batch_size), desc="Generating DPO candidates"):
            batch = rendered[start:start + batch_size]

            for sample_index, seed in enumerate(seeds):
                sampling_params = build_sampling_params(config, seed)
                outputs = llm.generate([item["input_text"] for item in batch], sampling_params)

                for item, output in zip(batch, outputs):
                    first_output = output.outputs[0]
                    row = build_generation_row(config, item, first_output, sample_index, seed)
                    generations_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Generations written to %s", generations_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DPO candidates with multiple seeds.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to infer DPO config")
    parser.add_argument("--run-id", type=str, help="override run_id from the json config")
    parser.add_argument("--model", "--student", dest="student", type=str, default=None, help="override models.student model path")
    parser.add_argument("--revision", type=str, default=None, help="override model/tokenizer revision")
    parser.add_argument("--seeds", type=str, help="comma-separated generation seeds, e.g. 9,19,29,39")
    parser.add_argument("--num-samples", type=int, help="number of random seeds to generate when --seeds is not provided")
    parser.add_argument("--temperature", type=float, help="override inference.temperature")
    parser.add_argument("--max-tokens", type=int, help="override inference.max_new_tokens")
    parser.add_argument("--max-model-len", type=int, help="override inference.max_model_len")
    parser.add_argument("--limit", type=int, help="override dataset.limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    generate(config)


if __name__ == "__main__":
    main()
