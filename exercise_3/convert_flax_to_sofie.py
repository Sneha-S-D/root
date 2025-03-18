import jax
import jax.numpy as jnp
from flax import nnx
import ROOT

# Define the FLAX model (from previous exploration)
class DeepMLP(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = nnx.Linear(2, 8, rngs=rngs)
        self.layer2 = nnx.Linear(8, 4, rngs=rngs)
        self.layer3 = nnx.Linear(4, 1, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)
        self.rngs = rngs

    def __call__(self, x, train=True):
        x = nnx.relu(self.layer1(x))
        x = self.dropout(x, deterministic=not train)
        x = nnx.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Converter class using PyROOT
class FlaxToSOFIEConverter:
    def __init__(self, flax_model):
        self.flax_model = flax_model
        # Access RModel from ROOT's TMVA namespace
        self.rmodel = ROOT.TMVA.Experimental.SOFIE.RModel()

    def convert(self):
        rngs = nnx.Rngs(0)
        flax_model = DeepMLP(rngs)
        for name, module in flax_model.__dict__.items():
            if isinstance(module, nnx.Linear):
                weights = module.weight.value
                bias = module.bias.value if module.bias is not None else None
                # Create ROperator (simplified)
                op = ROOT.TMVA.Experimental.SOFIE.ROperator()
                op.type = "Linear"
                op.input_shapes = [(-1, weights.shape[1])]  # Batch, input dim
                op.output_shapes = [(-1, weights.shape[0])]  # Batch, output dim
                op.attributes = {"weights": weights.tolist(), "bias": bias.tolist() if bias is not None else None}
                self.rmodel.AddOperator(op)
        return self.rmodel

# Convert and generate
if __name__ == "__main__":
    converter = FlaxToSOFIEConverter(None)  # flax_model not needed here initially
    rmodel = converter.convert()
    rmodel.Generate()
    rmodel.OutputGenerated("flax_model.hxx")
    print("Converted RModel generated to flax_model.hxx")
