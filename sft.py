import json
import argparse
import logging
import csv
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig


def build_messages(example, default_system_prompt=None, include_assistant=True):
    system_prompt = example.get("system") or default_system_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": example["instruction"]})
    if include_assistant:
        messages.append({"role": "assistant", "content": example["output"]})

    return messages


def make_tokenize_func(tokenizer, max_length, default_system_prompt=None):
    def tokenize_func(example):
        try:
            prompt_text = tokenizer.apply_chat_template(
                build_messages(example, default_system_prompt, include_assistant=False),
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                build_messages(example, default_system_prompt, include_assistant=True),
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            tokenized = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            labels = tokenized["input_ids"].copy()
            prompt_length = min(len(prompt_ids), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
            tokenized["labels"] = labels
            return tokenized
        except Exception as e:
            logging.warning(f"Error processing sample: {str(e)}")
            return {"input_ids": [], "attention_mask": [], "labels": []}

    return tokenize_func


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


def resolve_output_dir(config):
    if "run_id" not in config:
        return config["training"]["output_dir"]

    experiment_id = config.get("experiment_id", "TN02_sft")
    root_dir = Path(config.get("output_root", "experiments"))
    output_dir = root_dir / experiment_id / "runs" / config["run_id"]
    config["training"]["output_dir"] = str(output_dir)
    return config["training"]["output_dir"]


class MetricsHistoryCallback(TrainerCallback):
    CSV_FIELDS = [
        "step",
        "epoch",
        "loss",
        "train_loss",
        "mean_token_accuracy",
        "learning_rate",
        "grad_norm",
        "num_tokens",
    ]

    def __init__(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_path / "train_metrics_history.jsonl"
        self.csv_path = output_path / "train_metrics_history.csv"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        record = {
            "step": state.global_step,
            "epoch": logs.get("epoch", state.epoch),
            "loss": logs.get("loss"),
            "train_loss": logs.get("train_loss"),
            "mean_token_accuracy": logs.get("mean_token_accuracy"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "num_tokens": logs.get("num_input_tokens_seen"),
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        csv_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with open(self.csv_path, "a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.CSV_FIELDS)
            if not csv_exists:
                writer.writeheader()
            writer.writerow(record)


def train(config):
    dataset = load_dataset("json", data_files=config["dataset"]["labeled_path"])
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    # KV cache is for autoregressive generation; disable it during training.
    student_model.config.use_cache = False

    if student_tokenizer.chat_template is None:
        raise ValueError("Student tokenizer has no chat_template; cannot use apply_chat_template.")

    system_prompt = config["dataset"].get("system_prompt")
    resolve_output_dir(config)

    training_config = dict(config["training"])
    if training_config.pop("use_8bit_optimizer", False):
        if training_config.get("optim") not in (None, "adamw_bnb_8bit"):
            logging.warning(
                "use_8bit_optimizer=True overrides training.optim=%s to adamw_bnb_8bit",
                training_config["optim"],
            )
        training_config["optim"] = "adamw_bnb_8bit"

    dataset_kwargs = training_config.setdefault("dataset_kwargs", {})
    dataset_kwargs.setdefault("skip_prepare_dataset", True)
    config["training"] = training_config
    training_arguments = SFTConfig(**training_config)
    write_resolved_config(config, training_arguments.output_dir)

    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    limit = config["dataset"].get("limit")
    if limit is not None:
        dataset["train"] = dataset["train"].select(range(min(limit, len(dataset["train"]))))

    train_dataset = dataset["train"].map(
        make_tokenize_func(student_tokenizer, training_arguments.max_length, system_prompt),
        remove_columns=dataset["train"].column_names,
    )
    trainer = SFTTrainer(
        model=student_model,
        processing_class=student_tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        callbacks=[MetricsHistoryCallback(training_arguments.output_dir)],
    )
        
    trainer.train()
    trainer.save_model(training_arguments.output_dir)
    student_tokenizer.save_pretrained(training_arguments.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)  


if __name__ == "__main__":
    main()
