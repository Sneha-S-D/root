import jax
import jax.numpy as jnp
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, input_size, hidden_size, output_size, rngs):
        self.linear1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, output_size, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x

rngs = nnx.Rngs(0)
model = MLP(5, 16, 1, rngs)
x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
output = model(x)
print("FLAX Output:", output)
