# thesis-sft-0.6b

## SFT Commands

### Train SFT tu config
```bash
python sft.py --config <CONFIG_JSON>
```

Vi du config dang co trong repo:
- `configs/sft_0.6b_full.json`

## Eval Commands

### 1) Greedy benchmark generation
```bash
python eval_benchmark_greedy.py \
  --run-id <RUN_ID> \
  --student <MODEL_OR_PATH> \
  --revision <REVISION>
```

Vi du config benchmark:
- `configs/eval_benchmark_greedy.json`

### 2) Rule eval cho benchmark generations
```bash
python eval/rule_eval.py \
  --input <GENERATIONS_JSONL> \
  --label-field <LABEL_FIELD> \
  --pred-field <PRED_FIELD> \
  --question-field <QUESTION_FIELD> \
  --run-id-field <RUN_ID_FIELD> \
  --index-field <INDEX_FIELD>
```

### 3) Tao metrics va negative-case report
```bash
python eval/benchmark_metrics.py \
  --prediction <PREDICTION_JSONL> \
  --generations <GENERATIONS_JSONL> \
  --metrics-output <METRICS_MD_FILENAME> \
  --negative-output <NEGATIVE_MD_FILENAME>
```

### 4) Bulk rule eval (PowerShell)
```powershell
Get-ChildItem -Path <ROOT_DIR> -Recurse -Filter generations.jsonl -File |
  ForEach-Object {
    python .\eval\rule_eval.py --input $_.FullName
  }
```

### 5) Bulk metrics (PowerShell)
```powershell
Get-ChildItem -Path <ROOT_DIR> -Recurse -Filter prediction.jsonl -File |
  ForEach-Object {
    python .\eval\benchmark_metrics.py --prediction $_.FullName
  }
```

### 6) Negative analysis
```bash
python eval/negative_analysis.py \
  --prediction <PREDICTION_JSONL>
```

### 7) Positive analysis
```bash
python eval/positive_analysis.py \
  --prediction <PREDICTION_JSONL> \
  --generations <GENERATIONS_JSONL> \
  --dataset <DATASET_PARQUET>
```

### 8) Correct-token statistics
```bash
python eval/correct_token_stats.py \
  --prediction <PREDICTION_JSONL> \
  --output <OUTPUT_MD>
```

### 9) Export suspect false negatives
```bash
python eval/export_suspect_false_negative.py \
  --prediction <PREDICTION_JSONL> \
  --generations <GENERATIONS_JSONL> \
  --output-dir <OUTPUT_DIR> \
  --chunk-size <N>
```
