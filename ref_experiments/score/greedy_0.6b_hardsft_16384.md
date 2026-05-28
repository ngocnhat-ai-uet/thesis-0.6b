# Metrics By Dataset

- Prediction: `D:\my_workspace\math-reasoning-slm\score\greedy_0.6b_hardsft_16384\prediction.jsonl`
- Generations: `D:\my_workspace\math-reasoning-slm\score\greedy_0.6b_hardsft_16384\generations.jsonl`
- Total rows: 600

## Overall

- total: 600
- correct: 264 (44.00%)
- incorrect: 336 (56.00%)
- avg_output_token (correct): 2750.31
- avg_output_token (incorrect): 15028.21

## Dataset: aime24

- total: 30
- correct: 0 (0.00%)
- incorrect: 30 (100.00%)
- avg_output_token (correct): n/a
- avg_output_token (incorrect): 15604.20

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 2 | 0 | 2 | 4687.00 |
| length | 28 | 0 | 28 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 28 |
| reasoning_think | 2 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 28 |
| solution | 2 |

## Dataset: aime25

- total: 30
- correct: 2 (6.67%)
- incorrect: 28 (93.33%)
- avg_output_token (correct): 4368.00
- avg_output_token (incorrect): 16384.00

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 2 | 2 | 0 | 4368.00 |
| length | 28 | 0 | 28 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 27 |
| reasoning_think | 3 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 28 |
| solution | 2 |

## Dataset: amc23

- total: 40
- correct: 11 (27.50%)
- incorrect: 29 (72.50%)
- avg_output_token (correct): 3961.09
- avg_output_token (incorrect): 15927.14

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 12 | 11 | 1 | 3892.25 |
| length | 28 | 0 | 28 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 27 |
| reasoning_think | 13 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 28 |
| solution | 12 |

## Dataset: math

- total: 500
- correct: 251 (50.20%)
- incorrect: 249 (49.80%)
- avg_output_token (correct): 2684.35
- avg_output_token (incorrect): 14701.66

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 285 | 251 | 34 | 2848.86 |
| length | 215 | 0 | 215 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| reasoning_think | 292 |
| unclosed_think | 208 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| solution | 285 |
| none | 212 |
| thought | 3 |
