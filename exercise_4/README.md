# FLAX Model Parsing Exercise (Exercise 4)

This project implements a Python function to parse a FLAX model built with the `flax.linen` API and extract its configuration into a dictionary. The parsing function is designed to provide a detailed, structured representation of the model’s architecture, making it useful for debugging, documentation, or model analysis.


## Installation

Ensure you have Python 3.x installed, then set up the required dependencies in a virtual environment:


# Install dependencies
```bash
pip install flax jax
```

Example FLAX Model

Here’s a simple FLAX model to demonstrate the parsing function:
```bash
import flax.linen as nn

class SimpleModel(nn.Module):
    features: int  # Number of features for the first Dense layer

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)    # First dense layer
        x = nn.relu(x)                    # ReLU activation
        x = nn.Dense(self.features // 2)(x)  # Second dense layer with half the features
        return x
```

# The Parsing Function
The parse_flax_model function extracts the model’s configuration. Save this in a file named parse_flax_model.py:

```bash
import flax.linen as nn
import jax
import jax.numpy as jnp

def parse_flax_model(model: nn.Module) -> dict:
    config = {
        "features": model.features,  # Extract top-level attributes
        "layers": []
    }
    
    # Fake input to initialize the model and get layer details
    x = jnp.ones((1, 10))  # Assuming input shape (batch_size, input_dim)
    variables = model.init(jax.random.PRNGKey(0), x)
    params = variables["params"]
    
    # Extract layer information from the initialized parameters
    for i, (layer_name, layer_params) in enumerate(params.items()):
        layer_config = {
            "name": layer_name,
            "type": "Dense",  # Assuming all are Dense layers for this example
            "features": layer_params["kernel"].shape[-1],  # Output features
            "bias": "bias" in layer_params  # Check if bias is present
        }
        config["layers"].append(layer_config)
    
    return config
```

# Output
test_simple_model passed!
test_functional_model passed!

# Functional model configuration
```bash
{
    'features': 64,
    'layers': [
        {'name': 'Dense_0', 'type': 'Dense', 'features': 64, 'bias': True},
        {'name': 'Dense_1', 'type': 'Dense', 'features': 32, 'bias': True},
        {'name': 'Dense_2', 'type': 'Dense', 'features': 16, 'bias': True}
    ]
}
```


# How It Works
Attribute Extraction: Captures model-level attributes (e.g., features).

Model Initialization: Uses a fake input to initialize the model, revealing its layer structure.

Layer Parsing: Iterates through the initialized parameters to extract layer details (name, type, features, bias).

Output: Returns a dictionary with the model’s configuration.

