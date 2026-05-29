import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint directory")
    parser.add_argument("--output", required=True, help="Merged model output directory")
    parser.add_argument("--base-model", default=None, help="Override base model from adapter config")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)

    if not (checkpoint / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json found in {checkpoint}")

    with open(checkpoint / "adapter_config.json", "r", encoding="utf-8") as file:
        adapter_config = json.load(file)
    base_model = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("Missing base model. Pass --base-model or set base_model_name_or_path in adapter_config.json")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
