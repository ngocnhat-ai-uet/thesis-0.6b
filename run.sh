python3 0_eval_benchmark_greedy.py \
  --run-id benchmark_greedy_base_max16384_evalprompt_sys \
  --student Qwen/Qwen3-0.6B \
  --system-prompt-mode system

python3 0_eval_benchmark_greedy.py \
  --run-id benchmark_greedy_base_max16384_evalprompt_usersuffix \
  --student Qwen/Qwen3-0.6B \
  --system-prompt-mode user

python3 0_eval_benchmark_greedy.py \
  --run-id benchmark_greedy_sft9670_trainprompt_sys_max16384_evalprompt_sys \
  --student Qwen/Qwen3-0.6B \
  --system-prompt-mode system

python3 0_eval_benchmark_greedy.py \
  --run-id benchmark_greedy_sft9670_trainprompt_usersuffix_max16384_evalprompt_usersuffix \
  --student Qwen/Qwen3-0.6B \
  --system-prompt-mode user

python3 1_sft.py \
  --config configs/train_sft.json \
  --run-id sft9670_trainprompt_sys \
  --system-prompt-mode system

python3 1_sft.py \
  --config configs/train_sft.json \
  --run-id sft9670_trainprompt_usersufffix \
  --system-prompt-mode user

python3 1_sft_lora.py \
  --config configs/train_sft_lora.json \
  --run-id sft9670_lora_r128_alllinear_trainprompt_sys \
  --system-prompt-mode system

find score -type f -name "generations.jsonl" -exec python3 eval/rule_eval.py --input {} \;
find score -type f -name "prediction.jsonl" -exec sh -c 'python3 eval/benchmark_metrics.py --prediction "$1" --generations "$(dirname "$1")/generations.jsonl"' sh {} \;

python3 2_dpo_infer.py --config configs/infer_dpo.json --num-samples 4 --seeds 1,2,3,4
python3 2_dpo_infer.py --config configs/infer_dpo.json --seeds 1,2,3,4

python3 merge_lora_checkpoint.py \
  --checkpoint sft_experiments/sft9670_lora_r128_alllinear_trainprompt_sys/checkpoint-157 \
  --output sft_experiments/sft9670_lora_r128_alllinear_trainprompt_sys/merged-checkpoint-157
# Không cần chỉ rõ base-model, nó tự lấy base-model trong adapter_config.json

python3 0_eval_benchmark_greedy.py \
  --run-id benchmark_greedy_sft10000_lora_ep3_trainprompt_sys_max16384_evalprompt_sys \
  --student sft_experiments/sft9670_lora_r128_alllinear_trainprompt_sys/merged-checkpoint-471 \
  --system-prompt-mode system

python3 4_dpo_train.py --config configs/train_dpo.json --run-id train_dpo11_v2_m4096 --data-path data/dapo/train_dpo11_v2.json && \
python3 4_dpo_train.py --config configs/train_dpo.json --run-id train_dpo11_v3_m4096 --data-path data/dapo/train_dpo11_v3.json && \
python3 4_dpo_train.py --config configs/train_dpo.json --run-id train_dpo11_v4_m4096 --data-path data/dapo/train_dpo11_v4.json && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v2_m4096 --student dpo_experiments/train_dpo11_v2_m4096 && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v3_m4096 --student dpo_experiments/train_dpo11_v3_m4096 && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v4_m4096 --student dpo_experiments/train_dpo11_v4_m4096

python3 4_dpo_train.py --config configs/train_dpo.json --run-id train_dpo11_v6_m4096_b16 --data-path data/dapo/train_dpo11_v6.json --gradient-accumulation-steps 8 && \
python3 4_dpo_train.py --config configs/train_dpo.json --run-id train_dpo11_v5_m4096 --data-path data/dapo/train_dpo11_v5.json --gradient-accumulation-steps 32 && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v6_m4096_b16 --student dpo_experiments/train_dpo11_v6_m4096_b16 && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v5_m4096 --student dpo_experiments/train_dpo11_v5_m4096 && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v4_m4096 --student dpo_experiments/train_dpo11_v4_m4096 && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v3_m4096 --student dpo_experiments/train_dpo11_v3_m4096 && \
python3 0_eval_benchmark_greedy.py --run-id benchmark_greedy_train_dpo11_v2_m4096 --student dpo_experiments/train_dpo11_v2_m4096