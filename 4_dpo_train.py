import json
import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOTrainer, DPOConfig
import copy


DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_PROMPT_MODE_NONE = "none"
SYSTEM_PROMPT_MODE_SYSTEM = "system"
SYSTEM_PROMPT_MODE_USER = "user"
VALID_SYSTEM_PROMPT_MODES = {
    SYSTEM_PROMPT_MODE_NONE,
    SYSTEM_PROMPT_MODE_SYSTEM,
    SYSTEM_PROMPT_MODE_USER,
}


def build_messages(
    example,
    system_prompt=None,
    include_assistant=False,
    assistant_field=None,
    system_prompt_mode=SYSTEM_PROMPT_MODE_SYSTEM,
):
    messages = []
    user_text = example["prompt"]

    if system_prompt_mode == SYSTEM_PROMPT_MODE_SYSTEM and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif system_prompt_mode == SYSTEM_PROMPT_MODE_USER and system_prompt:
        user_text = f"{user_text}\n{system_prompt}"

    messages.append({"role": "user", "content": user_text})
    if include_assistant:
        messages.append({"role": "assistant", "content": example[assistant_field]})

    return messages


def process_dataset(
    dataset_path,
    dataset_seed,
    tokenizer,
    system_prompt=None,
    system_prompt_mode=SYSTEM_PROMPT_MODE_SYSTEM,
):
    examples = []
    with open(dataset_path, 'r') as file:
        examples = json.load(file)
    output_text = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    # use chat template
    for i in range(len(examples)):
        try:
            prompt = tokenizer.apply_chat_template(
                build_messages(
                    examples[i],
                    system_prompt,
                    include_assistant=False,
                    system_prompt_mode=system_prompt_mode,
                ),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            chosen_full = tokenizer.apply_chat_template(
                build_messages(
                    examples[i],
                    system_prompt,
                    include_assistant=True,
                    assistant_field="chosen",
                    system_prompt_mode=system_prompt_mode,
                ),
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            if not chosen_full.startswith(prompt):
                logging.warning("Skipping sample %s: chosen_full does not start with prompt.", i)
                continue
            chosen = chosen_full[len(prompt):]

            rejected_full = tokenizer.apply_chat_template(
                build_messages(
                    examples[i],
                    system_prompt,
                    include_assistant=True,
                    assistant_field="rejected",
                    system_prompt_mode=system_prompt_mode,
                ),
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            if not rejected_full.startswith(prompt):
                logging.warning("Skipping sample %s: rejected_full does not start with prompt.", i)
                continue
            rejected = rejected_full[len(prompt):]

            output_text["prompt"].append(prompt)
            output_text["chosen"].append(chosen)
            output_text["rejected"].append(rejected)
        except Exception as e:
            logging.warning(f"Error processing sample {i}: {str(e)}")
            
    dataset = Dataset.from_dict(output_text)
    dataset = dataset.shuffle(seed=dataset_seed)        
    return dataset


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
        raise ValueError("Missing required top-level config key: run_id")

    root_dir = Path(config.get("output_root", "dpo_experiments"))
    output_dir = root_dir / config["run_id"]
    config["training"]["output_dir"] = str(output_dir)
    return config["training"]["output_dir"]


class MetricsHistoryCallback(TrainerCallback):
    LOG_KEY_MAP = {
        "rewards_chosen": "rewards/chosen",
        "rewards_rejected": "rewards/rejected",
        "rewards_accuracies": "rewards/accuracies",
        "rewards_margins": "rewards/margins",
        "logps_chosen": "logps/chosen",
        "logps_rejected": "logps/rejected",
        "logits_chosen": "logits/chosen",
        "logits_rejected": "logits/rejected",
    }

    def __init__(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_path / "train_metrics_history.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        record = {
            "step": state.global_step,
            "epoch": logs.get("epoch", state.epoch),
            "loss": logs.get("loss"),
            "train_loss": logs.get("train_loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "num_tokens": logs.get("num_input_tokens_seen"),
        }
        for field, log_key in self.LOG_KEY_MAP.items():
            record[field] = logs.get(log_key)

        with open(self.jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(config):    
    dataset_path = config["dataset"]["labeled_path"]
    dataset_seed = config["dataset"]["seed"]

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if student_tokenizer.chat_template is None:
        raise ValueError("Student tokenizer has no chat_template; cannot use apply_chat_template.")

    system_prompt_mode = config["dataset"].get("system_prompt_mode", SYSTEM_PROMPT_MODE_SYSTEM)
    if system_prompt_mode not in VALID_SYSTEM_PROMPT_MODES:
        valid_modes = ", ".join(sorted(VALID_SYSTEM_PROMPT_MODES))
        raise ValueError(
            f"Invalid dataset.system_prompt_mode={system_prompt_mode!r}. "
            f"Valid values: {valid_modes}"
        )

    system_prompt = config["dataset"].get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    dataset = process_dataset(
        dataset_path,
        dataset_seed,
        student_tokenizer,
        system_prompt,
        system_prompt_mode,
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )    
    # KV cache is for autoregressive generation; disable it during training.
    student_model.config.use_cache = False
    if student_model.config.pad_token_id is None:
        student_model.config.pad_token_id = student_tokenizer.pad_token_id

    resolve_output_dir(config)

    training_config = dict(config["training"])
    if training_config.pop("use_8bit_optimizer", False):
        if training_config.get("optim") not in (None, "adamw_bnb_8bit"):
            logging.warning(
                "use_8bit_optimizer=True overrides training.optim=%s to adamw_bnb_8bit",
                training_config["optim"],
            )
        training_config["optim"] = "adamw_bnb_8bit"

    config["training"] = training_config
    max_len = training_config.pop("max_len", None)
    if max_len is not None:
        training_config["max_length"] = max_len
    training_config.pop("scheduler_specific_kwargs", None)
    training_config["lr_scheduler_type"] = "cosine_with_min_lr"
    training_config["lr_scheduler_kwargs"] = {
        "min_lr": training_config["learning_rate"] * 0.1,
    }
    training_arguments = DPOConfig(**training_config)
    write_resolved_config(config, training_arguments.output_dir)

    trainer = DPOTrainer(
        student_model,
        ref_model=copy.deepcopy(student_model),
        args=training_arguments,
        train_dataset=dataset,
        processing_class=student_tokenizer,
        callbacks=[MetricsHistoryCallback(training_arguments.output_dir)],
    )

    trainer.train()
    trainer.save_model(training_arguments.output_dir)
    student_tokenizer.save_pretrained(training_arguments.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    parser.add_argument('--run-id', type=str, help='override run_id from the json config')
    parser.add_argument('--data-path', type=str, help='override dataset.labeled_path from the json config')
    parser.add_argument('--gradient-accumulation-steps', type=int, help='override training.gradient_accumulation_steps from the json config')
    args = parser.parse_args()
    config = json.load(open(args.config))
    if args.run_id:
        config["run_id"] = args.run_id
    if args.data_path:
        config.setdefault("dataset", {})["labeled_path"] = args.data_path
    if args.gradient_accumulation_steps is not None:
        config.setdefault("training", {})["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    train(config)


if __name__ == "__main__":
    main()
