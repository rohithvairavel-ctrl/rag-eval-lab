# Cross-Validation
K-fold cross-validation splits data into K folds, training on K-1 and validating on the held-out fold, rotating until each fold is used once. Average metrics estimate generalization.
Stratified K-fold preserves class proportions — preferred for classification imbalance.
Leave-one-out (LOO) uses N folds for small datasets but is computationally expensive.
Time-series CV must respect temporal order (forward chaining). Nested CV separates model selection from final evaluation to avoid optimistic bias.
