#!/usr/bin/env bash
set -euo pipefail

FULL_DATASET="${FULL_DATASET:-math_dapo_processed.jsonl}"
DATASET="${DATASET:-dpo_data/dpo_split/overeasy_candidate/medium.jsonl}"
INFER_BASE_CONFIG="${INFER_BASE_CONFIG:-dpo_configs/infer_dpo.json}"
TRAIN_BASE_CONFIG="${TRAIN_BASE_CONFIG:-dpo_configs/train_dpo.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-dpo_experiments}"
START_LEVEL="${START_LEVEL:-11}"
END_LEVEL="${END_LEVEL:-16}"
PREVIOUS_LEVEL=$((START_LEVEL - 1))
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
DPO_MODE="${DPO_MODE:-normal}"

LEVELS=(11 12 13 14 15 16)
TEMPS=(0.7 0.7 0.7 1.0 1.0 1.2)

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "${FULL_DATASET}" "full dataset jsonl"
require_path "${DATASET}" "medium dataset"
require_path "${INFER_BASE_CONFIG}" "infer base config"
require_path "${TRAIN_BASE_CONFIG}" "train base config"
if [[ -e "${MODEL}" ]]; then
  :
elif [[ "${MODEL}" == Qwen/* ]]; then
  echo "Using remote initial model: ${MODEL}"
else
  echo "Missing initial model: ${MODEL}" >&2
  exit 1
fi

mkdir -p dpo_data/dpo_train

for i in "${!LEVELS[@]}"; do
  LEVEL="${LEVELS[$i]}"
  TEMP="${TEMPS[$i]}"

  if (( LEVEL < START_LEVEL || LEVEL > END_LEVEL )); then
    continue
  fi

  INFER_RUN="infer_dpo${LEVEL}_qwen3_0.6b"
  TRAIN_RUN="train_dpo${LEVEL}_qwen3_0.6b"

  INFER_DIR="${OUTPUT_ROOT}/${INFER_RUN}"
  GENERATIONS="${INFER_DIR}/generations.jsonl"
  PREDICTION="${INFER_DIR}/prediction.jsonl"
  TRAIN_JSON="dpo_data/dpo_train/${TRAIN_RUN}.json"
  INFER_CONFIG="/tmp/${INFER_RUN}.json"
  TRAIN_CONFIG="/tmp/${TRAIN_RUN}.json"

  echo "===== LEVEL ${LEVEL}: infer model=${MODEL}, temp=${TEMP}, dataset=${DATASET} ====="

  env \
    INFER_BASE_CONFIG="${INFER_BASE_CONFIG}" \
    INFER_CONFIG="${INFER_CONFIG}" \
    INFER_RUN="${INFER_RUN}" \
    DATASET="${DATASET}" \
    MODEL="${MODEL}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    TEMP="${TEMP}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    python3 -c '
import json, os

with open(os.environ["INFER_BASE_CONFIG"], "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["run_id"] = os.environ["INFER_RUN"]
cfg["output_root"] = os.environ["OUTPUT_ROOT"]
cfg.setdefault("dataset", {})["data_path"] = os.environ["DATASET"]
cfg.setdefault("models", {})["student"] = os.environ["MODEL"]
cfg.setdefault("inference", {})["temperature"] = float(os.environ["TEMP"])
cfg["inference"]["num_samples"] = int(os.environ["NUM_SAMPLES"])
cfg["inference"]["max_new_tokens"] = int(os.environ["MAX_TOKENS"])
cfg["inference"]["max_model_len"] = int(os.environ["MAX_MODEL_LEN"])

with open(os.environ["INFER_CONFIG"], "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
'

  python3 14_dpo_infer.py \
    --config "${INFER_CONFIG}" \
    --run-id "${INFER_RUN}" \
    --model "${MODEL}" \
    --num-samples "${NUM_SAMPLES}" \
    --temperature "${TEMP}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}"

  python3 eval/dpo_rule_eval.py \
    --input "${GENERATIONS}" \
    --enable-symbolic

  python3 15_dpo_build_dataset.py build-dpo \
    --mode "${DPO_MODE}" \
    --dataset "${DATASET}" \
    --prediction "${PREDICTION}" \
    --generations "${GENERATIONS}" \
    --output "${TRAIN_JSON}"

  env \
    TRAIN_BASE_CONFIG="${TRAIN_BASE_CONFIG}" \
    TRAIN_CONFIG="${TRAIN_CONFIG}" \
    TRAIN_RUN="${TRAIN_RUN}" \
    TRAIN_JSON="${TRAIN_JSON}" \
    MODEL="${MODEL}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    python3 -c '
import json, os

with open(os.environ["TRAIN_BASE_CONFIG"], "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["run_id"] = os.environ["TRAIN_RUN"]
cfg["output_root"] = os.environ["OUTPUT_ROOT"]
cfg.setdefault("dataset", {})["labeled_path"] = os.environ["TRAIN_JSON"]
cfg.setdefault("models", {})["student"] = os.environ["MODEL"]

with open(os.environ["TRAIN_CONFIG"], "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
'

  python3 16_dpo_train.py \
    --config "${TRAIN_CONFIG}"

  MODEL="${OUTPUT_ROOT}/${TRAIN_RUN}"
  require_path "${MODEL}" "trained model for next level"

  echo "===== DONE LEVEL ${LEVEL}: next model=${MODEL} ====="
done

echo "Final model: ${MODEL}"
