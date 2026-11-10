# SmartLight AI - results summary

Generated automatically by `run_pipeline.py`. All test-set numbers come from data the
model never saw during training, tuning or model selection.

## Data
- Rows in: 34,310; usable: 34,310; assets: 94
- Removed: missing 0, unknown weather 0, out of range 0, duplicates 0
- Class counts: {'No Fault': 24111, 'Short Circuit': 1706, 'Voltage Surge': 1720, 'Bulb Failure': 3359, 'Light Flickering': 3414}
- Majority-class share: 70.3% (accuracy of always guessing 'No Fault')

## Model comparison (validation set)
| model | fit_seconds | val_accuracy | val_balanced_accuracy | val_macro_f1 | val_mcc | val_bulb_recall | val_bulb_precision |
|---|---|---|---|---|---|---|---|
| Majority baseline | 0.100 | 0.703 | 0.200 | 0.165 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.100 | 0.458 | 0.648 | 0.437 | 0.339 | 0.613 | 0.233 |
| Random Forest | 1.200 | 0.824 | 0.614 | 0.688 | 0.595 | 0.190 | 0.419 |
| Random Forest + augmentation | 6.900 | 0.844 | 0.609 | 0.687 | 0.652 | 0.135 | 0.971 |
| SVM (RBF) | 61.100 | 0.826 | 0.553 | 0.631 | 0.608 | 0.062 | 1.000 |
| Gradient Boosting | 11.000 | 0.662 | 0.659 | 0.611 | 0.433 | 0.567 | 0.239 |
| Neural Network (MLP) | 22.500 | 0.836 | 0.605 | 0.676 | 0.627 | 0.143 | 0.774 |

## Cross-validation (train + validation)
| model | cv_folds | cv_accuracy_mean | cv_accuracy_std | cv_macro_f1_mean | cv_macro_f1_std |
|---|---|---|---|---|---|
| Logistic Regression | 3 | 0.468 | 0.003 | 0.448 | 0.003 |
| Random Forest | 3 | 0.826 | 0.002 | 0.692 | 0.005 |
| Gradient Boosting | 3 | 0.685 | 0.003 | 0.624 | 0.003 |
| Neural Network (MLP) | 3 | 0.836 | 0.002 | 0.674 | 0.005 |

## Selected model
- **Random Forest + augmentation** (highest MCC among models within 0.005 of best validation macro-F1)
- Bulb-failure threshold: none (argmax)

## Test-set performance (final model)
- Accuracy **0.8438**, balanced accuracy 0.6118, macro-F1 **0.6868**, MCC 0.6522
- Majority-class baseline: accuracy 0.7027, macro-F1 0.1651
- Same model with plain argmax (no bulb threshold): accuracy 0.8438, macro-F1 0.6868

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| No Fault | 0.820 | 0.997 | 0.900 | 3617 |
| Short Circuit | 0.995 | 0.727 | 0.840 | 256 |
| Voltage Surge | 0.950 | 0.516 | 0.668 | 258 |
| Bulb Failure | 0.932 | 0.109 | 0.195 | 504 |
| Light Flickering | 1.000 | 0.711 | 0.831 | 512 |

## Bulb-failure threshold sweep (validation)
| threshold | rule | accuracy | macro_f1 | bulb_precision | bulb_recall | bulb_f1 |
|---|---|---|---|---|---|---|
| nan | argmax | 0.844 | 0.687 | 0.971 | 0.135 | 0.237 |
| 0.100 | bulb threshold | 0.628 | 0.668 | 0.207 | 0.817 | 0.330 |
| 0.150 | bulb threshold | 0.700 | 0.684 | 0.229 | 0.637 | 0.337 |
| 0.200 | bulb threshold | 0.758 | 0.689 | 0.248 | 0.415 | 0.310 |
| 0.250 | bulb threshold | 0.804 | 0.694 | 0.318 | 0.278 | 0.297 |
| 0.300 | bulb threshold | 0.826 | 0.691 | 0.426 | 0.194 | 0.267 |
| 0.350 | bulb threshold | 0.838 | 0.689 | 0.645 | 0.155 | 0.250 |
| 0.400 | bulb threshold | 0.842 | 0.687 | 0.831 | 0.137 | 0.235 |
| 0.450 | bulb threshold | 0.844 | 0.688 | 0.945 | 0.137 | 0.239 |
| 0.500 | bulb threshold | 0.844 | 0.688 | 0.986 | 0.135 | 0.237 |
| 0.550 | bulb threshold | 0.844 | 0.688 | 1.000 | 0.135 | 0.238 |
| 0.600 | bulb threshold | 0.844 | 0.688 | 1.000 | 0.135 | 0.238 |

## Time-series features experiment (random forest, validation)
- 30 lag / delta / rolling features added
- macro-F1 0.6883 -> 0.6905 (+0.0021); temporal context did not help meaningfully; the single-reading model is kept

## Chronological hold-out
- accuracy 0.8366, macro-F1 0.6847 (trained on the earliest 70% of time, tested on the latest 15%)

## Anomaly detection (test set, 'is this reading faulty?')
| detector | roc_auc | precision | recall | f1 | false_positive_rate | detects_Short Circuit | detects_Voltage Surge | detects_Bulb Failure | detects_Light Flickering |
|---|---|---|---|---|---|---|---|---|---|
| Baseline envelope | 0.741 | 0.960 | 0.486 | 0.645 | 0.009 | 0.738 | 0.512 | 0.113 | 0.713 |
| Isolation Forest | 0.655 | 0.574 | 0.073 | 0.130 | 0.023 | 0.125 | 0.062 | 0.024 | 0.102 |
| Autoencoder | 0.610 | 0.713 | 0.118 | 0.203 | 0.020 | 0.355 | 0.012 | 0.014 | 0.156 |
| Deployed (envelope OR both ML agree) | nan | 0.940 | 0.488 | 0.643 | 0.013 | 0.738 | 0.512 | 0.117 | 0.717 |

## Feature importance (permutation, validation)
| feature | importance_mean | importance_std |
|---|---|---|
| current_fluctuations (Amperes) | 0.165 | 0.006 |
| current_fluctuations_env (Amperes) | 0.158 | 0.003 |
| voltage_levels (Volts) | 0.118 | 0.003 |
| power_consumption (Watts) | 0.039 | 0.002 |
| environmental_conditions | 0.005 | 0.004 |
| temperature (Celsius) | -0.002 | 0.001 |

## Time-series analysis
- Span 2023-01-01 00:33:30 to 2023-12-31 23:46:21; 94 assets; median 365 readings per asset
- EWMA z-score anomaly flags vs labelled faults: {'signal': 'ANY sensor', 'flagged': 1351, 'precision': 0.9015544041450777, 'recall': 0.12119402985074627, 'f1': 0.21366546794140867}
- One-step forecast MAE (voltage_levels (Volts)): EWMA 8.122189586214276, naive 10.380518343133941
- Fault onsets found for pre-failure analysis: 3484
- Top risk assets: [{'asset_id': 'ASSET-SL-2026-086', 'risk_score': 0.3508, 'risk_level': 'High'}, {'asset_id': 'ASSET-SL-2026-033', 'risk_score': 0.3168, 'risk_level': 'High'}, {'asset_id': 'ASSET-SL-2026-057', 'risk_score': 0.3136, 'risk_level': 'High'}, {'asset_id': 'ASSET-SL-2026-075', 'risk_score': 0.297, 'risk_level': 'Medium'}, {'asset_id': 'ASSET-SL-2026-069', 'risk_score': 0.2959, 'risk_level': 'Medium'}]

## Chatbot
- Intent classifier: 97.4% accuracy on 39 held-out phrasings (trained on 113)

## Tickets (generated from the test set)
- 750 tickets; 318 dispatch-now; 14 need verification; signatures valid: True

Model file: 10.25 MB. Runtime: 232 s. scikit-learn 1.9.1.