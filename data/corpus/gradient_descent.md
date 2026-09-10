# Gradient Descent
Gradient descent minimizes a loss by iteratively moving parameters opposite the gradient. Learning rate controls step size.
Batch GD uses the full dataset per step (stable, slow). Stochastic GD uses one example (noisy, fast). Mini-batch is the practical default.
Momentum, Adam, and RMSProp adapt step sizes and accelerate convergence. Learning rate schedules (step decay, cosine) improve late-stage training.
Vanishing/exploding gradients are mitigated by careful init, normalization, and residual connections.
