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