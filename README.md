# TrainConvnetusingPrimeNumbers

A novel approach to training convolutional neural networks using prime number-based sampling for enhanced generalization.

## Overview

This repository contains a PyTorch implementation of a CNN training method that uses prime numbers to create varying subset samplers for each training epoch. The approach allows the model to see different subsets of the training data in each epoch, potentially improving generalization and reducing overfitting.

## Key Features

- Uses ResNet18 architecture with transfer learning
- Implements a unique prime number-based sampling strategy
- Creates different subsets of training data for each epoch
- Includes comprehensive data augmentation techniques
- Optimized for GPU acceleration

## Prime Number Sampling Strategy

The training process employs a novel approach:

1. Generate a list of prime numbers less than the training dataset size
2. For each epoch, select a prime number and create a subset of indices that are divisible by that prime
3. Train the model using this subset, cycling through different primes across epochs

This strategy creates dynamic sampling patterns, where:
- Small primes (2, 3, 5, 7) → larger subsets for stable initial learning
- Medium primes (11-31) → moderate subsets for refining features
- Large primes (37+) → smaller, highly specific subsets for fine-tuning

## Training Progression

### Initial Learning Phase (Small Primes)
- Using primes 2, 3, 5, 7 → larger data samples
- Validation accuracy rapidly increases from 0.7255 to 0.9020

### Intermediate Learning Phase (Medium Primes)
- Using primes 11-31 → moderate sample sizes
- Training accuracy shows controlled variability while trending upward
- Validation accuracy stabilizes between 0.88-0.92

### Final Learning Phase (Large Primes)
- Using primes 37+ → smaller, more focused samples
- Training accuracy reaches 1.0 in most cases
- Validation accuracy maintains above 0.88

The varying subset sizes help prevent overfitting while maintaining high performance, with a final best validation accuracy of 0.921569.

## Requirements

- PyTorch
- torchvision
- CUDA-capable GPU (recommended)
- NumPy
- Matplotlib
- PIL

## Dataset

The code is configured to work with the Hymenoptera dataset (ants vs. bees classification), but can be adapted for other image classification tasks.

## Usage

1. Update the `data_dir` variable to point to your dataset location
2. Adjust hyperparameters as needed
3. Run the script to train the model
4. The final model will be saved as "residueNorm_resnet18_9.pth"

## Citation

If you use this code or method in your research, please cite:

```
@misc{bemoregt2025primesampling,
  author = {bemoregt},
  title = {Training Convolutional Networks using Prime Numbers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bemoregt/TrainConvnetusingPrimeNumbers}
}
```