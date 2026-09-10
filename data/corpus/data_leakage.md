# Data Leakage
Leakage is using information that would not be available at prediction time, inflating offline metrics.
Examples: scaling before split, target-derived features, future timestamps, and duplicate near-identical rows across splits.
Prevent by pipeline discipline: fit transformers inside CV folds, time-aware splits, and leakage audits.
If test AUC looks too good to be true, check for leakage first.
