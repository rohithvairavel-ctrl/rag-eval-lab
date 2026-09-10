# Handling Imbalanced Data
When one class dominates, accuracy is misleading. Prefer precision, recall, F1, PR-AUC, or cost-sensitive metrics.
Techniques: class weights, oversampling (SMOTE), undersampling, threshold tuning, and anomaly/rare-event formulations.
Collecting more minority examples or better features often beats resampling tricks.
Evaluate with stratified splits and report confusion matrices, not accuracy alone.
