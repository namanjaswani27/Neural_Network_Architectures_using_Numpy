# Implementing Recurrent Neural Networks from Scratch

## Background
The task was to train a RNN model (using `NUMPY`) to generate text similar to Harry-Potter literature(Fiction). So this was a Character-Level language generation task.

## Experiment
- Preprocessed the raw text to remove unwanted characters
- Built character level vocabulary and indexed each unique character in the entire corpus
- Defined inputs and targets for our language model. Basically used a windowed approach to take `n-1` characters as input and the `nth` character as target for this input.
- Tokenized inputs and targets to feed them to model
- Trained a RNN model with Adam optimizer and gradient clipping
- Softmax activation was used for output at each timestep
- At inference, given a small piece of text, the model would generate the rest tokens in a greedy way.

```
PATH = '../input/harry-potter/'
  MAX_LEN = 100
  GRAD_MIN = -5
  GRAD_MAX = 5
  HIDDEN_SIZE = 256
  LR = 3e-4
  TRAIN_SIZE_FRACTION = 0.9
  EPOCHS = 5
  TEMPERATURE = 0.5

  SEED = 42

  BETA1 = 0.9
  BETA2 = 0.999
  EPS = 1e-8
  USE_ADAM = True
  ```
  
  > Kndly refer to the attached report(pdf) for a detailed analysis
