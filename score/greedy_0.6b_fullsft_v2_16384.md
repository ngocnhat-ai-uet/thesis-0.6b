# Metrics By Dataset

- Prediction: `D:\my_workspace\math-reasoning-slm\_score\greedy_0.6b_fullsft_v2_16384_ep3\prediction.jsonl`
- Generations: `D:\my_workspace\math-reasoning-slm\_score\greedy_0.6b_fullsft_v2_16384_ep3\generations.jsonl`
- Total rows: 600

## Overall

- total: 600
- correct: 275 (45.83%)
- incorrect: 325 (54.17%)
- avg_output_token (correct): 2579.43
- avg_output_token (incorrect): 14542.24

## Dataset: aime24

- total: 30
- correct: 1 (3.33%)
- incorrect: 29 (96.67%)
- avg_output_token (correct): 4572.00
- avg_output_token (incorrect): 15314.31

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 4 | 1 | 3 | 5675.75 |
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

## Dataset: aime25

- total: 30
- correct: 3 (10.00%)
- incorrect: 27 (90.00%)
- avg_output_token (correct): 4547.33
- avg_output_token (incorrect): 15493.70

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 6 | 3 | 3 | 6459.33 |
| length | 24 | 0 | 24 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 24 |
| reasoning_think | 6 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 24 |
| solution | 6 |

## Dataset: amc23

- total: 40
- correct: 9 (22.50%)
- incorrect: 31 (77.50%)
- avg_output_token (correct): 1994.11
- avg_output_token (incorrect): 14415.77

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 16 | 9 | 7 | 4476.25 |
| length | 24 | 0 | 24 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| unclosed_think | 24 |
| reasoning_think | 16 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| none | 23 |
| solution | 16 |
| thought | 1 |

## Dataset: math

- total: 500
- correct: 262 (52.40%)
- incorrect: 238 (47.60%)
- avg_output_token (correct): 2569.40
- avg_output_token (incorrect): 14356.70

### Finish Reason

| finish_reason | count | correct | incorrect | avg_output_token |
| --- | ---: | ---: | ---: | ---: |
| stop | 304 | 262 | 42 | 2890.83 |
| length | 196 | 0 | 196 | 16384.00 |

### Think Type (count)

| think_type | count |
| --- | ---: |
| reasoning_think | 304 |
| unclosed_think | 196 |

### Last Box Source (count)

| last_box_source | count |
| --- | ---: |
| solution | 304 |
| none | 194 |
| thought | 2 |
