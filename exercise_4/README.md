# FLAX Model Parsing Exercise

This project implements a Python function to parse a FLAX model and extract its configuration into a dictionary. The function is designed to work with FLAX models built using the `flax.linen` API.

---

## **What Does This Do?**

The `parse_flax_model` function takes a FLAX model as input and returns a dictionary containing:
1. **Model attributes** (e.g., `features`).
2. **Layer information** (e.g., layer type, number of features, bias).

This is useful for understanding and documenting the structure of a FLAX model.

---

## **How to Use**

### **Step 1: Install Dependencies**
Make sure you have the required libraries installed:
```bash
pip install flax jax

## **example of a simple FLAX model**:
import flax.linen as nn

class SimpleModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 2)(x)
        return x

## **Import the parse_flax_model function and use it to extract the model configuration**:
from parse_flax_model import parse_flax_model

model = SimpleModel(features=64)
config = parse_flax_model(model)
print(config)


## **Expected Output**:
{
    'features': 64,
    'layers': [
        {'name': 'Dense_0', 'type': 'Dense', 'features': 64, 'bias': True},
        {'name': 'Dense_1', 'type': 'Dense', 'features': 32, 'bias': True}
    ]
}

## **testing**:
**Test 1: Simple Model**

run the test script for the simple model:
python test_parse_flax_model.py

**Expected Output**
test_simple_model passed!

**Test 2: Functional Model**

class FunctionalModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 4)(x)
        return x

run the test script:
python test_parse_flax_model.py

**Expected Output**
test_simple_model passed!
test_functional_model passed!

**Functional Model Configuration output**
{
    'features': 64,
    'layers': [
        {'name': 'Dense_0', 'type': 'Dense', 'features': 64, 'bias': True},
        {'name': 'Dense_1', 'type': 'Dense', 'features': 32, 'bias': True},
        {'name': 'Dense_2', 'type': 'Dense', 'features': 16, 'bias': True}
    ]
}


How It Works

The parse_flax_model function:

Extracts model attributes (e.g., features).
Initializes the model with fake input to access its parameters.
Extracts layer information (e.g., layer type, number of features, bias).
Returns a dictionary with all the details.

Dependencies

Python 3.x
FLAX (pip install flax)
JAX (pip install jax)

