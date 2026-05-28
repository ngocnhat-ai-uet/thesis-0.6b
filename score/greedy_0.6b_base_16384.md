# Metrics By Dataset

- Prediction: `D:\my_workspace\math-reasoning-slm\score\greedy_0.6b_base_16384\prediction.jsonl`
- Generations: `D:\my_workspace\math-reasoning-slm\score\greedy_0.6b_base_16384\generations.jsonl`
- Total rows: 600

## Overall

- total: 600
- correct: 374 (62.33%)
- incorrect: 226 (37.67%)
- avg_output_token (correct): 2568.57
- avg_output_token (incorrect): 12314.73

## Dataset: aime24

- total: 30
- correct: 1 (3.33%)
- incorrect: 29 (96.67%)
- avg_output_token (correct): 3607.00
- avg_output_token (incorrect): 14350.41

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 7 | 1 | 6 | 6133.86 |
| length | 23 | 0 | 23 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 22 |
| reasoning_think | 8 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 21 |
| solution | 7 |
| thought | 2 |

## Dataset: aime25

- total: 30
- correct: 4 (13.33%)
- incorrect: 26 (86.67%)
- avg_output_token (correct): 4270.00
- avg_output_token (incorrect): 13643.54

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 13 | 4 | 9 | 7175.69 |
| length | 17 | 0 | 17 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 16 |
| reasoning_think | 14 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 16 |
| solution | 13 |
| thought | 1 |

## Dataset: amc23

- total: 40
- correct: 17 (42.50%)
- incorrect: 23 (57.50%)
- avg_output_token (correct): 4192.53
- avg_output_token (incorrect): 13041.00

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 24 | 17 | 7 | 4544.67 |
| length | 16 | 0 | 16 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| reasoning_think | 24 |
| unclosed_think | 16 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| solution | 24 |
| none | 16 |

## Dataset: math

- total: 500
- correct: 352 (70.40%)
- incorrect: 148 (29.60%)
- avg_output_token (correct): 2467.86
- avg_output_token (incorrect): 11569.54

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 428 | 351 | 77 | 3274.13 |
| length | 72 | 1 | 71 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| reasoning_think | 429 |
| unclosed_think | 71 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| solution | 428 |
| none | 67 |
| thought | 5 |
