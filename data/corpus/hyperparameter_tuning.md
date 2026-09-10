# Hyperparameter Tuning
Grid search exhaustively tries combinations; random search often finds good configs faster in high dimensions.
Bayesian optimization (TPE, Gaussian processes) models the objective to pick promising trials. Early stopping / Hyperband allocate budget adaptively.
Tune on validation folds; keep a final holdout for unbiased reporting. Log all trials for reproducibility.
Important knobs: learning rate, regularization, depth/width, batch size, and embedding dim.
