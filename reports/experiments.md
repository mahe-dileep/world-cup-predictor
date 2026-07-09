# Results

Held-out test set: **2019–2025, 6,868 matches** (strict temporal split).

## Baselines vs models

| Model | Accuracy | Log loss |
| --- | --- | --- |
| Majority class (always Home) | 0.479 | 1.050 |
| Higher-Elo pick / Elo-only logit | 0.600 | 0.868 |
| Softmax engine (from scratch) | 0.611 | 0.862 |
| CatBoost (production model) | 0.612 | 0.859 |

## Experiment 1 — does squad market value help?

Softmax on Elo + form vs the same + `squad_value_diff`, on matches between the 48 finalists (6,209 train / 787 test). Note: market values are a *current* snapshot used as a static team attribute.

| Features | Accuracy | Log loss |
| --- | --- | --- |
| Elo + form | 0.497 | 1.018 |
| + squad_value_diff | 0.490 | 1.014 |

**Delta: -0.006 accuracy, -0.003 log loss** (negative log loss = better).

## Experiment 2 — weighted-K vs flat-K Elo (tournament matches)

Logistic on Elo difference, trained on non-World-Cup matches, evaluated on all 1,049 World Cup finals matches.

| Elo variant | Accuracy | Log loss |
| --- | --- | --- |
| Weighted-K (goal margin) | 0.558 | 0.971 |
| Flat-K | 0.556 | 0.979 |

**Delta (weighted − flat): +0.002 accuracy, -0.008 log loss.**
