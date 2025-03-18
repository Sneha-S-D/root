import jax.numpy as jnp
from parse_flax_model import SimpleModel, parse_flax_model

def test_simple_model():
    model = SimpleModel(features=64)
    config = parse_flax_model(model)
    
    assert config['features'] == 64
    assert len(config['layers']) == 2
    assert config['layers'][0]['type'] == 'Dense'
    assert config['layers'][0]['features'] == 64
    assert config['layers'][1]['type'] == 'Dense'
    assert config['layers'][1]['features'] == 32
    print("test_simple_model passed!")

if __name__ == "__main__":
    test_simple_model()
