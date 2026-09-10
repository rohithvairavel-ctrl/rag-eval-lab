# Precision, Recall, and F1
Precision = TP / (TP + FP): of predicted positives, how many are correct. High precision means few false alarms.
Recall (sensitivity) = TP / (TP + FN): of actual positives, how many were found. High recall means few misses.
F1 is the harmonic mean of precision and recall: 2PR/(P+R). Useful when classes are imbalanced and both errors matter.
Accuracy can mislead on imbalanced data. Choose the metric based on business cost of FP vs FN.
