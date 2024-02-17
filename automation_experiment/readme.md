# Automation Experiment

This is an experiment to begin to automate the learning. First, the output of the trained model needs to be returned so it can be persisted. This will allow several successive runs to be analysed and an attempt to discover patterns in the variables changed from run to run.

Some of the hyperparameters that can be changed:

1. Number of layers
2. Learning rate
3. Activation Functions (with appropriate kernel initializer)
4. Number of units per Dense function
5. the layers that go between each Dense layer (BatchNormalization, Dropout, Conv2D + Flatten)
   5.a additional tuning hyperparameter ranges

A training run produces a correct result with high confidence, or an incorrect result with high confidence. Correct results are summed, while incorrect results are subtracted, producing a single result. Highly positive numbers are correlated to highly confident, correctly determined hyperparameter settings, while lower or negative numbers correspond to fewer correct hyperparameter settings.

When several runs' test results are looked at together alongside hyperparameter settings a pattern must emerge, similar to a sentence.
