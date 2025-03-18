import jax
import jax.numpy as jnp
from flax import nnx
import optax

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

rngs = nnx.Rngs(0)
model = DeepMLP(rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        pred = model(x, train=True)
        return jnp.mean((pred - y) ** 2)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
y = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
loss = train_step(model, optimizer, x, y)
print("Initial loss:", loss)
output = model(x, train=False)
print("Inference output:", output)
