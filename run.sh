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


find score -type f -name "generations.jsonl" -exec python3 eval/rule_eval.py --input {} \;
find score -type f -name "prediction.jsonl" -exec sh -c 'python3 eval/benchmark_metrics.py --prediction "$1" --generations "$(dirname "$1")/generations.jsonl"' sh {} \;

python3 2_dpo_infer.py --config configs/infer_dpo.json --num-samples 4 --seeds 1,2,3,4
python3 2_dpo_infer.py --config configs/infer_dpo.json --seeds 1,2,3,4