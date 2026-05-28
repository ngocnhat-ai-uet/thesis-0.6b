import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONFIG_PATH = Path("configs/eval_benchmark_greedy.json")
MAX_NEW_TOKENS = 16384
DEFAULT_SYSTEM_PROMPT = r"Solve the math problem step by step. Put the final answer in \\boxed{}."
SYSTEM_PROMPT_MODE_NONE = "none"
SYSTEM_PROMPT_MODE_SYSTEM = "system"
SYSTEM_PROMPT_MODE_USER = "user"
VALID_SYSTEM_PROMPT_MODES = {
    SYSTEM_PROMPT_MODE_NONE,
    SYSTEM_PROMPT_MODE_SYSTEM,
    SYSTEM_PROMPT_MODE_USER,
}


# Input: dataset_path (jsonl/parquet)
# Output: list of dict
def load_records(dataset_config):
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


def get_question(record):
    for field in ("question", "instruction", "prompt"):
        if field in record and record[field] is not None:
            return str(record[field])
    raise ValueError(f"Record has no question/instruction/prompt field: {record}")


def get_index(record, fallback_index):
    for field in ("index", "id", "question_idx"):
        if field in record and record[field] is not None:
            return record[field]
    return fallback_index


def get_dataset_name(record):
    dataset = record.get("dataset")
    if dataset is None:
        return "unknown"
    return str(dataset)


def build_prompt_text(question):
    return f"{question}"


def build_messages(question, system_prompt=None, system_prompt_mode=SYSTEM_PROMPT_MODE_SYSTEM):
    user_text = build_prompt_text(question)
    messages = []

    if system_prompt_mode == SYSTEM_PROMPT_MODE_SYSTEM and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif system_prompt_mode == SYSTEM_PROMPT_MODE_USER and system_prompt:
        user_text = f"{user_text}\n{system_prompt}"

    messages.append({"role": "user", "content": user_text})
    return messages


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    config.setdefault("inference", {})["max_new_tokens"] = MAX_NEW_TOKENS
    return config


def apply_cli_overrides(config, args):
    if args.run_id:
        config["run_id"] = args.run_id
    if args.student:
        config.setdefault("models", {})["student"] = args.student
    if args.revision:
        config.setdefault("models", {})["revision"] = args.revision
    return config


def load_tokenizer_and_vllm(config):
    model_path = config["models"].get("student") or config["models"].get("model")
    model_revision = config["models"].get("revision")
    if not model_path:
        raise ValueError("Config must define models.student or models.model")

    logging.info(f"Loading ckpt and tokenizer: {model_path}")
    if model_revision:
        logging.info(f"Using model/tokenizer revision: {model_revision}")
    tokenizer_kwargs = {"trust_remote_code": True}
    if model_revision:
        tokenizer_kwargs["revision"] = model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "left"

    if tokenizer.eos_token is None:
        raise ValueError("No available eos_token.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    logging.info(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")

    num_gpus = torch.cuda.device_count()
    attention_backend = config["inference"].get("attention_backend")
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
        logging.info(f"Using vLLM attention backend: {attention_backend}")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=num_gpus,
        enable_chunked_prefill=config["inference"]["enable_chunked_prefill"],
        gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
        trust_remote_code=config["inference"]["trust_remote_code"],
        dtype=torch.bfloat16,
        enforce_eager=config["inference"]["enforce_eager"],
        max_model_len=config["inference"]["max_model_len"],
    )
    if model_revision:
        llm_kwargs["revision"] = model_revision
    llm = LLM(**llm_kwargs)
    logging.info("vLLM model loaded successfully")
    return tokenizer, llm


def render_inputs(records, config, tokenizer):
    dataset_config = config.setdefault("dataset", {})
    system_prompt_mode = dataset_config.get("system_prompt_mode", SYSTEM_PROMPT_MODE_SYSTEM)
    if system_prompt_mode not in VALID_SYSTEM_PROMPT_MODES:
        valid_modes = ", ".join(sorted(VALID_SYSTEM_PROMPT_MODES))
        raise ValueError(
            f"Invalid dataset.system_prompt_mode={system_prompt_mode!r}. "
            f"Valid values: {valid_modes}"
        )
    system_prompt = dataset_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    rendered = []
    for index, record in enumerate(records):
        question = get_question(record)
        record_index = get_index(record, index)
        messages = build_messages(question, system_prompt, system_prompt_mode)
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["inference"].get("enable_thinking", True),
        )
        rendered.append(
            {
                "dataset": get_dataset_name(record),
                "index": record_index,
                "question": question,
                "label": record.get("answer", record.get("final_answer", record.get("label"))),
                "input_text": full_text,
            }
        )
    return rendered


def prepare_run_dir(config):
    run_id = config["run_id"]
    root_dir = Path(config.get("output_root", "score"))
    run_dir = root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(config, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_path = output_path / "config.resolved.yaml"
    try:
        import yaml

        with open(resolved_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    except Exception:
        with open(resolved_path, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)


def build_sampling_params(config):
    return SamplingParams(
        n=1,
        # top_k=config["inference"].get("top_k", 1),
        top_p=config["inference"].get("top_p", 1.0),
        min_p=config["inference"].get("min_p", 0.0),
        temperature=config["inference"]["temperature"],
        presence_penalty=config["inference"].get("presence_penalty", 0.0),
        seed=config["inference"]["seed"],
        skip_special_tokens=False,
        ignore_eos=False,
        max_tokens=MAX_NEW_TOKENS,
        stop=["<|im_end|>"]
    )


def get_limited_records(config):
    records = load_records(config["dataset"])
    limit = config["dataset"].get("limit")
    if limit is not None:
        records = records[: int(limit)]
    return records


def generate(config):
    records = get_limited_records(config)
    tokenizer, llm = load_tokenizer_and_vllm(config)
    rendered = render_inputs(records, config, tokenizer)

    run_dir = prepare_run_dir(config)
    write_resolved_config(config, run_dir)
    generations_path = run_dir / "generations.jsonl"
    batch_size = config["inference"].get("batch_size", 32)
    sampling_params = build_sampling_params(config)

    logging.info("Starting benchmark generation with max_new_tokens=%d", MAX_NEW_TOKENS)
    with open(generations_path, "w", encoding="utf-8") as output_file:
        for start in tqdm(range(0, len(rendered), batch_size), desc="Generating responses"):
            batch = rendered[start:start + batch_size]
            outputs = llm.generate([item["input_text"] for item in batch], sampling_params)

            for item, output in zip(batch, outputs):
                first_output = output.outputs[0]
                row = {
                    "run_id": config["run_id"],
                    "dataset": item["dataset"],
                    "index": item["index"],
                    "question": item["question"],
                    "label": item["label"],
                    "model_output": first_output.text,
                    "output_token_length": len(getattr(first_output, "token_ids", []) or []),
                    "input_text": item["input_text"],
                    "finish_reason": getattr(first_output, "finish_reason", None),
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Generations written to %s", generations_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, help="benchmark run id")
    parser.add_argument("--student", type=str, help="student model path")
    parser.add_argument("--revision", type=str, help="model/tokenizer revision")
    args = parser.parse_args()

    config = load_config()
    config = apply_cli_overrides(config, args)
    generate(config)


if __name__ == "__main__":
    main()
