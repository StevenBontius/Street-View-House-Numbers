## Baseline

A baseline run with default settings showed clear overfitting:

- num_filters = 32
- num_layers = 3  
- num_hidden_units = 128
- learning_rate = 0.001
- batch_size = 32

Train loss dropped to 0.06 while test loss increased to 0.47, with best test accuracy of ~91% at epoch 5. Early stopping triggered after 5 epochs without improvement.

## Hypothesis

Since the model is already overfitting with the baseline architecture, increasing model complexity is unlikely to help. A simpler architecture with fewer layers or filters may generalize better by reducing excess capacity.

Experiment 1 will sweep num_layers (2-4), num_filters (16-128) and num_hidden_units (64-512) to find the optimal architecture before applying regularization.

## Search space

- `num_layers`: 3 options (2, 3, 4)
- `num_filters`: 4 options (16, 32, 64, 128)
- `num_hidden_units`: 4 options (64, 128, 256, 512)

Total: 3 × 4 × 4 = 48 combinations

## Experiment 1 Results

20 random search trials were run, covering 42% of the search space.

[bubble chart here]

The bubble chart shows that model complexity does not improve generalization. Higher complexity consistently leads to larger overfitting gaps and worse test loss. The sweet spot lies in low to medium complexity models.

[num_layers vs num_filters heatmap here]

Accuracy ranges narrowly from 0.84 to 0.91 across all trials. 2 layers performs consistently worse, while 3 and 4 layers show similar results. No clear winner emerges for num_filters.

All 20 trials show a positive loss gap (test loss - train loss), confirming that overfitting persists regardless of architecture choice. Trial 16 (4 layers, 16 filters, 256 hidden units) achieved the best balance: 91.0% accuracy with the smallest overfitting gap of 0.107.

## Hypothesis Experiment 2

Architecture variation alone does not resolve overfitting. Trial 16 is selected as the base architecture for experiment 2, which will investigate whether dropout and batchnorm can reduce the overfitting gap while maintaining accuracy.