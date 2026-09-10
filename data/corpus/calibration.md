# Probability Calibration
A classifier is calibrated if predicted probabilities match empirical frequencies (e.g., 70% predictions are correct ~70% of the time).
Reliability diagrams visualize calibration. Brier score and ECE quantify miscalibration.
Platt scaling and isotonic regression recalibrate scores post-hoc. Temperature scaling is common for neural nets.
Good ranking (high AUC) does not imply good calibration.
