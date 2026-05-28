# Metrics By Dataset

- Prediction: `D:\my_workspace\math-reasoning-slm\score\greedy_0.6b_fullsft_16384\prediction.jsonl`
- Generations: `D:\my_workspace\math-reasoning-slm\score\greedy_0.6b_fullsft_16384\generations.jsonl`
- Total rows: 600

## Overall

- total: 600
- correct: 263 (43.83%)
- incorrect: 337 (56.17%)
- avg_output_token (correct): 2912.35
- avg_output_token (incorrect): 15375.40

## Dataset: aime24

- total: 30
- correct: 1 (3.33%)
- incorrect: 29 (96.67%)
- avg_output_token (correct): 12295.00
- avg_output_token (incorrect): 16384.00

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 1 | 1 | 0 | 12295.00 |
| length | 29 | 0 | 29 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 29 |
| reasoning_think | 1 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 29 |
| solution | 1 |

## Dataset: aime25

- total: 30
- correct: 2 (6.67%)
- incorrect: 28 (93.33%)
- avg_output_token (correct): 11070.50
- avg_output_token (incorrect): 15589.82

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 4 | 2 | 2 | 8168.00 |
| length | 26 | 0 | 26 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 26 |
| reasoning_think | 4 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 26 |
| solution | 4 |

## Dataset: amc23

- total: 40
- correct: 10 (25.00%)
- incorrect: 30 (75.00%)
- avg_output_token (correct): 3002.30
- avg_output_token (incorrect): 15519.03

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 12 | 10 | 2 | 3070.17 |
| length | 28 | 0 | 28 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 28 |
| reasoning_think | 12 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 28 |
| solution | 12 |

## Dataset: math

- total: 500
- correct: 250 (50.00%)
- incorrect: 250 (50.00%)
- avg_output_token (correct): 2805.96
- avg_output_token (incorrect): 15217.15

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 276 | 250 | 26 | 3028.12 |
| length | 224 | 0 | 224 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| reasoning_think | 280 |
| unclosed_think | 220 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| solution | 276 |
| none | 222 |
| thought | 2 |
