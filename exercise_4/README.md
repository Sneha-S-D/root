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
from typing import Tuple, Any

def parse_flax_model(model: nn.Module, input_shape: Tuple[int, ...]) -> dict:
    config = {}
    
    # Extract model attributes ( features, etc.)
    for attr, value in vars(model).items():
        if not attr.startswith('_'):  
            config[attr] = value
    
    # Initialize the model with the provided input shape
    rng = jax.random.PRNGKey(0)
    x = jnp.ones(input_shape)
    variables = model.init(rng, x)
    params = variables['params']
    
    # Use jax.eval_shape to get the overall output shape
    def apply_fn(params, x):
        return model.apply({'params': params}, x)
    output_shape_struct = jax.eval_shape(apply_fn, params, x)
    output_tensor_shape = output_shape_struct.shape
    
    # Extract layer information with tensor shapes
    layers = []
    current_input_shape = input_shape
    
    for name, param_dict in params.items():
        layer_info = {
            'name': name,
            'type': name.split('_')[0],  # Extract layer type 
            'features': param_dict['kernel'].shape[-1],  # Number of output features
            'bias': 'bias' in param_dict,
            'input_tensor_name': f"{name}/input",
            'input_tensor_shape': tuple(current_input_shape),
            'output_tensor_name': f"{name}/output"
        }
        
        # Compute output shape manually for Dense layers
        if layer_info['type'] == 'Dense':
            output_shape = (current_input_shape[0], param_dict['kernel'].shape[-1])
            layer_info['output_tensor_shape'] = output_shape
            current_input_shape = output_shape
        
        layers.append(layer_info)
    
    config['layers'] = layers
    config['input_shape'] = tuple(input_shape)
    config['output_shape'] = tuple(output_tensor_shape)
    return config```

# Output
test_simple_model passed!
test_functional_model passed!

# Simple model configuration
```bash
SimpleModel Config: {
  "features": 64,
  "name": null,
  "layers": [
    {
      "name": "Dense_0",
      "type": "Dense",
      "features": 64,
      "bias": true,
      "input_tensor_name": "Dense_0/input",
      "input_tensor_shape": [
        1,
        10
      ],
      "output_tensor_name": "Dense_0/output",
      "output_tensor_shape": [
        1,
        64
      ]
    },
    {
      "name": "Dense_1",
      "type": "Dense",
      "features": 32,
      "bias": true,
      "input_tensor_name": "Dense_1/input",
      "input_tensor_shape": [
        1,
        64
      ],
      "output_tensor_name": "Dense_1/output",
      "output_tensor_shape": [
        1,
        32
      ]
    }
  ],
  "input_shape": [
    1,
    10
  ],
  "output_shape": [
    1,
    32
  ]
}


```

# Functional model configuration
```bash
FunctionalModel Config: {
  "features": 64,
  "name": null,
  "layers": [
    {
      "name": "Dense_0",
      "type": "Dense",
      "features": 64,
      "bias": true,
      "input_tensor_name": "Dense_0/input",
      "input_tensor_shape": [
        1,
        10
      ],
      "output_tensor_name": "Dense_0/output",
      "output_tensor_shape": [
        1,
        64
      ]
    },
    {
      "name": "Dense_1",
      "type": "Dense",
      "features": 32,
      "bias": true,
      "input_tensor_name": "Dense_1/input",
      "input_tensor_shape": [
        1,
        64
      ],
      "output_tensor_name": "Dense_1/output",
      "output_tensor_shape": [
        1,
        32
      ]
    }
  ],
  "input_shape": [
    1,
    10
  ],
  "output_shape": [
    1,
    32
  ]
}
```


# How It Works
Attribute Extraction: Captures model-level attributes (e.g., features).

Model Initialization: Uses a fake input to initialize the model, revealing its layer structure.

Layer Parsing: Iterates through the initialized parameters to extract layer details (name, type, features, bias).

Output: Returns a dictionary with the model’s configuration.

