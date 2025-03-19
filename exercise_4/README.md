# FLAX Model Parsing Exercise

This project implements a Python function to parse a FLAX model and extract its configuration into a dictionary. The function is designed to work with FLAX models built using the \`flax.linen\` API.

---

## **Install Dependencies**

Make sure you have the required libraries installed:

bash
pip install flax jax

---

## **Example of a Simple FLAX Model**

Here’s an example of a simple FLAX model:

\`\`\`python
import flax.linen as nn

class SimpleModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 2)(x)
        return x
\`\`\`

---

## **Using the Parsing Function**

Import the \`parse_flax_model\` function and use it to extract the model configuration:

\`\`\`python
from parse_flax_model import parse_flax_model

model = SimpleModel(features=64)
config = parse_flax_model(model)
print(config)
\`\`\`

---

## **Expected Output**

The output will be a dictionary describing the model:

\`\`\`python
{
    'features': 64,
    'layers': [
        {'name': 'Dense_0', 'type': 'Dense', 'features': 64, 'bias': True},
        {'name': 'Dense_1', 'type': 'Dense', 'features': 32, 'bias': True}
    ]
}
\`\`\`

---

## **Testing**

### **Test 1: Simple Model**

Run the test script for the simple model:

\`\`\`bash
python test_parse_flax_model.py
\`\`\`

#### **Expected Output**

\`\`\`
test_simple_model passed!
\`\`\`

---

### **Test 2: Functional Model**

Here’s an example of a functional FLAX model:

\`\`\`python
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
\`\`\`

Run the test script:

bash
python test_parse_flax_model.py

#### **Expected Output**

\`\`\`
test_simple_model passed!
test_functional_model passed!
\`\`\`

#### **Functional Model Configuration Output**

python
{
    'features': 64,
    'layers': [
        {'name': 'Dense_0', 'type': 'Dense', 'features': 64, 'bias': True},
        {'name': 'Dense_1', 'type': 'Dense', 'features': 32, 'bias': True},
        {'name': 'Dense_2', 'type': 'Dense', 'features': 16, 'bias': True}
    ]
}


---

## **How It Works**

The \`parse_flax_model\` function:

1. **Extracts model attributes** (e.g., \`features\`).
2. **Initializes the model** with fake input to access its parameters.
3. **Extracts layer information** (e.g., layer type, number of features, bias).
4. **Returns a dictionary** with all the details.

---

## **Dependencies**

- Python 3.x
- FLAX (\`pip install flax\`)
- JAX (\`pip install jax\`)

---

