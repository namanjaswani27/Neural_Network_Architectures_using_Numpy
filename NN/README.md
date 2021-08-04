# Feed Forward Neural Network from scratch.

## Background
Given some training images with labels (happy/sad), the task was to learn a MLP model to predict labels for test images.

## Experimentation
- Flattened the images to a single vector (dim: 10201)
- Apply PCA to reduce the dimensionality (reduced dim: 12)
- Created a Single Layers class with forward and backward prop functions
- Final Model was built by stacking multiple Single layer networks

```
lr = 5e-1               # Learning Rate
input_size = 12         # Input dimensions
hidden_size = 15        # Hidden layer dimensions [# of nodes]
output_size = 2         # Output dimensions
eps = 1e-40             # Small value to prevent error in log
epochs = 20             # Number of iterations of backprop
random_state = 6        # Random_State to reproduce results
gamma = 0               # Momentum parameter
reg = 1e-2              # L2 Regularization parameter
```

Accuracy scores of 80% was achieved using 2 hidden layers
