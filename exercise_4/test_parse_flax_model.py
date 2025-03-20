from parse_flax_model import SimpleModel, FunctionalModel, parse_flax_model

def test_simple_model():
    model = SimpleModel(features=64)
    config = parse_flax_model(model, input_shape=(1, 10))
    
    assert config['features'] == 64
    assert len(config['layers']) == 2
    assert config['layers'][0]['type'] == 'Dense'
    assert config['layers'][0]['features'] == 64
    assert config['layers'][0]['input_tensor_shape'] == (1, 10)
    assert config['layers'][0]['output_tensor_shape'] == (1, 64)
    assert config['layers'][1]['type'] == 'Dense'
    assert config['layers'][1]['features'] == 32
    assert config['layers'][1]['input_tensor_shape'] == (1, 64)
    assert config['layers'][1]['output_tensor_shape'] == (1, 32)
    assert config['input_shape'] == (1, 10)
    assert config['output_shape'] == (1, 32)
    print("test_simple_model passed!")

def test_functional_model():
    model = FunctionalModel(features=64)
    config = parse_flax_model(model, input_shape=(1, 10))
    
    assert config['features'] == 64
    assert len(config['layers']) == 2
    assert config['layers'][0]['type'] == 'Dense'
    assert config['layers'][0]['features'] == 64
    assert config['layers'][0]['input_tensor_shape'] == (1, 10)
    assert config['layers'][0]['output_tensor_shape'] == (1, 64)
    assert config['layers'][1]['type'] == 'Dense'
    assert config['layers'][1]['features'] == 32
    assert config['layers'][1]['input_tensor_shape'] == (1, 64)
    assert config['layers'][1]['output_tensor_shape'] == (1, 32)
    assert config['input_shape'] == (1, 10)
    assert config['output_shape'] == (1, 32)
    print("test_functional_model passed!")

if __name__ == "__main__":
    test_simple_model()
    test_functional_model()
