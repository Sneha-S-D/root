import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple, Any

def parse_flax_model(model: nn.Module, input_shape: Tuple[int, ...]) -> dict:
    """
    Parses a FLAX model and returns its configuration as a dictionary.

    Args:
        model: A FLAX model (subclass of nn.Module).
        input_shape: Tuple representing the input shape (e.g., (batch_size, features) or (batch_size, height, width, channels)).

    Returns:
        A dictionary containing the model configuration with layer details and tensor shapes.
    """
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
    return config

# a simple FLAX model for testing
class SimpleModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 2)(x)
        return x

# model using the functional API
class FunctionalModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 2)(x)
        return x

# Test the function
if __name__ == "__main__":
    # Test with SimpleModel
    model = SimpleModel(features=64)
    config = parse_flax_model(model, input_shape=(1, 10))
    import json
    print("SimpleModel Config:", json.dumps(config, indent=2))

    # Test with FunctionalModel
    model = FunctionalModel(features=64)
    config = parse_flax_model(model, input_shape=(1, 10))
    print("FunctionalModel Config:", json.dumps(config, indent=2))

