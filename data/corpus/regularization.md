# L1 and L2 Regularization
L2 (Ridge) adds the squared magnitude of weights to the loss, shrinking coefficients toward zero without eliminating them. It handles multicollinearity well.
L1 (Lasso) adds the absolute value of weights, encouraging sparsity — many coefficients become exactly zero, useful for feature selection.
Elastic Net combines L1 and L2. The regularization strength is controlled by a hyperparameter λ (or alpha).
In neural nets, weight decay is typically L2 on parameters.
