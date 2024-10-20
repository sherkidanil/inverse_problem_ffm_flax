import os
import time
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax import linen as nn
from flax.training import train_state

import pickle
from scipy.integrate import solve_ivp


savedir = "models/simple_64_another_2_without_noise"
os.makedirs(savedir, exist_ok=True)

import logging
logging.basicConfig(filename=f'{savedir}/logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(jax.default_backend())

# 2. Train
class MLP(nn.Module):
    dim: int
    out_dim: int = 1
    w: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.out_dim)(x)
        return x
model = MLP(dim=4)

@jax.jit
def predict(params, inputs):
    return model.apply({"params": params}, inputs)

@jax.jit
def sample_conditional_pt(x0, x1, t, sigma):
    t = t.reshape(-1, *([1] * (x0.ndim - 1)))
    mu_t = t * x1 + (1 - t) * x0
    # epsilon = jax.random.normal(key, x0.shape)
    epsilon = 0
    return mu_t + sigma * epsilon

@jax.jit
def compute_conditional_vector_field(x0, x1):
    return x1 - x0

@jax.jit
def loss_ffm_function(params, x1, x0, d, e):
    t = jax.random.uniform(key, (x0.shape[0],))
    xt = sample_conditional_pt(x0, x1, t, 0.01)
    ut = compute_conditional_vector_field(x0, x1)
    inputs = jnp.concatenate([xt, d, e, t[:, None]], axis=-1)
    vt = predict(params, inputs)
    # loss = jnp.mean(jnp.power(jnp.linalg.norm(vt - ut),2))
    loss = jnp.mean((vt - ut) ** 2)
    return loss

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
batch_size = 128
num_epochs = 200000
learning_rate = 0.001
optimizer = optax.adamw(learning_rate=learning_rate)
params = model.init(key, jnp.ones((1, 4)))
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params["params"],
    tx=optimizer
)

losses = []

start = time.time()
for k in tqdm(range(num_epochs)):
    key, subkey = jax.random.split(subkey)
    m_key, e_key, noise_key = random.split(key, 3)
    m = jax.random.uniform(m_key, shape=(batch_size, 1))
    e = jax.random.uniform(e_key, shape=(batch_size, 1))
    noise = jax.random.normal(noise_key, shape=(batch_size, 1)) * 1e-4
    d = jnp.power(e, 2) * jnp.power(m, 3) + m * jnp.exp(-jnp.abs(0.2 - e)) + noise
    
    x0 = jax.random.uniform(subkey, (batch_size, 1))

    loss, grads = jax.value_and_grad(loss_ffm_function, has_aux=False)(state.params, m, x0, d, e)
    state = update_model(state, grads)
    losses.append(loss.item())

    if (k+1) % 1000 == 0:
        end = time.time()
        logging.info(f"{k+1}: loss {loss:0.3f} time {(end - start):0.2f}")


# 3. Saving
np.save(f"{savedir}/losses.npy", np.array(losses))

with open(f'{savedir}/w.pkl', 'wb') as f:
    pickle.dump(state.params, f)

# 4. Inference

def ode_function(t, m, d, e):
    inputs = jnp.array([m[0], d, e, t])
    return predict(state.params, inputs)[0]

def d_by_m_e(m, e):
    noise = np.random.normal(scale=1e-4, size=1).item()
    d = np.power(e, 2) * np.power(m, 3) + m * np.exp(-np.abs(0.2 - e)) + noise
    return d

errors, m_sol, d_err = [], [], []

for _ in tqdm(range(1000)):
    m0 = np.random.uniform(size=1).item()
    m = 0.2
    e = 0.1 
    d = d_by_m_e(m, e)

    solution = solve_ivp(ode_function, t_span=[0, 1], y0=[m0], t_eval=None, args=(d, e))
    m_sol.append(solution.y[0][-1])
    errors.append(np.abs(m_sol[-1] - m))
    d_err.append(np.abs(d_by_m_e(m_sol[-1], e) - d))

np.save(f'{savedir}/results.npy', np.array([errors, m_sol, d_err]))
logging.info(f'errors: {np.mean(errors)}, {np.std(errors)}')
logging.info(f'm: {np.mean(m_sol)}, {np.std(m_sol)}')
logging.info(f'd err: {np.mean(d_err)}, {np.std(d_err)}')
