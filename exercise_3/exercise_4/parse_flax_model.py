import flax.linen as nn
import jax
import jax.numpy as jnp

def parse_flax_model(model):
    """
    Parses a FLAX model and returns its configuration as a dictionary.

    Args:
        model: A FLAX model (subclass of nn.Module).

    Returns:
        A dictionary containing the model configuration.
    """
    config = {}
    
    # Extract model attributes 
    for attr, value in vars(model).items():
        if not attr.startswith('_'): 
            config[attr] = value
    
    # Initialize the model to access its parameters
    rng = jax.random.PRNGKey(0)
    input_shape = (1, 10)  
    x = jnp.ones(input_shape)
    variables = model.init(rng, x)
    
    # Extract layer information
    layers = []
    for name, params in variables['params'].items():
        layer_info = {
            'name': name,
            'type': name.split('_')[0],  # Extract layer type (e.g., 'Dense')
            'features': params['kernel'].shape[-1],  # Number of output features
            'bias': 'bias' in params  # Check if bias is present
        }
        layers.append(layer_info)
    
    config['layers'] = layers
    return config

# Define a simple FLAX model for testing
class SimpleModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 2)(x)
        return x

# Test the function
if __name__ == "__main__":
    model = SimpleModel(features=64)
    config = parse_flax_model(model)
    print("SimpleModel Config:", config)

# Define a model using the functional API
class FunctionalModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features // 2)(x)
        return x

# Test the functional API model
if __name__ == "__main__":
    # Test with SimpleModel
    model = SimpleModel(features=64)
    config = parse_flax_model(model)
    print("SimpleModel Config:", config)

    # Test with FunctionalModel
    model = FunctionalModel(features=64)
    config = parse_flax_model(model)
    print("FunctionalModel Config:", config)
