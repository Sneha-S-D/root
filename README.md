# GSoC 2025 Exercise 3: SOFIE Keras Parser and FLAX Exploration

## Overview
This directory contains solutions for Exercise 3 of the GSoC 2025 project TMVA SOFIE - Enhancing Keras Parser and JAX/FLAX Integration with CERN-HSF.

## Files
- `create_keras_models.py`: Generates a sample Keras model.
- `parse_keras_models.C`: Parses the Keras model using SOFIE.
- `flax_example.py`: Demonstrates a neural network using FLAX NNX.

## Observations
- Successfully built ROOT 6.35.01 with TMVA and SOFIE.
- Keras parser handled Dense layers; further testing needed for Conv2D.
- FLAX NNX uses a modular, stateful design with JAX integration.

## Next Steps
- Add tests for parser robustness.
- Explore FLAX-to-ONNX conversion for SOFIE integration.
