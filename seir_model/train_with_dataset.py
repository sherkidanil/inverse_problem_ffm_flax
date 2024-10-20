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

from typing import List

def d_by_m_e(m: List[int], e: List[int]):
    beta1, alpha, gamma_r, gamma_d1, beta2, gamma_d2 = m
    tau = 2.1
    
    def beta(t):
        return beta1 + 0.5 * np.tanh(7 * (t - tau)) * (beta2 - beta1)

    def gamma_d(t):
        return gamma_d1 + 0.5 * np.tanh(7 * (t - tau)) * (gamma_d2 - gamma_d1)

    def gamma(t):
        return gamma_r + gamma_d(t)

    def seir_model(t, y, beta, alpha, gamma):
        S, E, I, R = y
        dSdt = -beta(t) * S * I
        dEdt = beta(t) * S * I - alpha * E
        dIdt = alpha * E - gamma(t) * I
        dRdt = (gamma_r + gamma_d(t)) * I
        return [dSdt, dEdt, dIdt, dRdt]

    S0, E0, I0, R0 = 99, 1, 0, 0
    y0 = [S0, E0, I0, R0]

    solution = solve_ivp(seir_model, t_span=[0,4], y0=y0, t_eval=e, args=(beta, alpha, gamma))
    return solution.y[2:]


savedir = "models/seir_4p_ds_without_noise_2kk"
datedir = "data/seir_data"

os.makedirs(savedir, exist_ok=True)

import logging
logging.basicConfig(filename=f'{savedir}/logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(jax.default_backend())


# 1. Generation
m = jnp.load(f'{datedir}/m.npy')
d = jnp.load(f'{datedir}/d.npy')
e = jnp.load(f'{datedir}/e.npy')

# 2. Train
class MLP(nn.Module):
    dim: int
    out_dim: int = 1
    w: int = 512

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
model = MLP(dim=19, out_dim=6)

@jax.jit
def predict(params, inputs):
    return model.apply({"params": params}, inputs)

@jax.jit
def sample_conditional_pt(x0, x1, t, sigma):
    t = t.reshape(-1, *([1] * (x0.ndim - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = jax.random.normal(key, x0.shape)
    # epsilon = 0
    return mu_t + sigma * epsilon

@jax.jit
def compute_conditional_vector_field(x0, x1):
    return x1 - x0

@jax.jit
def loss_ffm_function(params, x1, x0, d, e, key):
    t = jax.random.uniform(key, (x0.shape[0],))
    xt = sample_conditional_pt(x0, x1, t, sigma=0.01)
    ut = compute_conditional_vector_field(x0, x1)
    inputs = jnp.concatenate([xt, d, e, t[:, None]], axis=-1)
    vt = predict(params, inputs)
    loss = jnp.mean((vt - ut) ** 2)
    return loss

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


key = jax.random.PRNGKey(0)
batch_size = 128
num_epochs = 3_000_000
learning_rate = 0.001
optimizer = optax.lion(learning_rate=learning_rate)
params = model.init(key, jnp.ones((1, 19)))
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params["params"],
    tx=optimizer
)

# X = jnp.load('data.npy')
dataset = jnp.concatenate([m, e, d], axis = 1)
loader = jax.random.permutation(key, dataset)

losses = []

start = time.time()
for k in tqdm(range(num_epochs)):
    key, subkey = jax.random.split(key)
    batch_indices = jax.random.choice(subkey, jnp.arange(len(dataset)), (batch_size,))
    batch = dataset[batch_indices]

    x0 = jax.random.uniform(subkey, (batch_size, 6))
    x1 = batch[:, :6]
    d = batch[:, 10:]
    e = batch[:, 6:10]

    loss, grads = jax.value_and_grad(loss_ffm_function, has_aux=False)(state.params, x1, x0, d, e, subkey)
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
m0 =np.random.uniform(size=6)
m = [0.4, 0.3, 0.3, 0.1, 0.15, 0.6]
e = np.linspace(1,3,4)
d = d_by_m_e(m ,e).flatten()
m0 = jnp.array(m0).reshape(1,-1)
m = jnp.array(m).reshape(1,-1)
e = jnp.array(e).reshape(1,-1)
d = jnp.array(d).reshape(1,-1)
dim = m[0].shape[0] + e[0].shape[0] + d[0].shape[0]
print(dim)

def ode_function(t, m, d, e):
    m = m.reshape(1, -1)
    t = jnp.array(t).reshape(1,-1)
    inputs = jnp.concatenate([m, d, e, t], axis=1)
    return predict(state.params, inputs)[0]


solution = solve_ivp(ode_function, t_span=[0, 1], y0=m0[0], t_eval=None, args=(d, e))
d_pred = d_by_m_e(solution.y[:, -1],e[0]).flatten()
diff_norm = jnp.linalg.norm(d - d_pred) / jnp.linalg.norm(d)

print(f'm_pred = {solution.y[:, -1]}')
print(f'd = {d}')
print(f'd_pred = {d_pred}')
print(f'diff norm = {diff_norm}')

logging.info(f'm_pred = {solution.y[:, -1]}')
logging.info(f'd = {d}')
logging.info(f'd_pred = {d_pred}')
logging.info(f'diff norm = {diff_norm}')

errors = []
sols = []

for i in tqdm(range(1000)):
    m = [0.4, 0.3, 0.3, 0.1, 0.15, 0.6]
    e = np.linspace(1,3,4)
    d = d_by_m_e(m ,e).flatten()
    m0 = jnp.array(np.random.uniform(size=6)).reshape(1,-1)
    m = jnp.array(m).reshape(1,-1)
    e = jnp.array(e).reshape(1,-1)
    d = jnp.array(d).reshape(1,-1)
    solution = solve_ivp(ode_function, t_span=[0, 1], y0=m0[0], t_eval=None, args=(d, e))
    d_pred = d_by_m_e(solution.y[:, -1],e[0]).flatten()
    sols.append(solution.y[:, -1])
    try:
        errors.append(np.linalg.norm(d - d_pred) / np.linalg.norm(d))
    except ValueError:
        pass

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

df = pd.DataFrame(sols, columns = [fr'$\beta_1$', fr'$\alpha$', fr'$\gamma^r$', fr'$\gamma^d_1$', fr'$\beta_2$', fr'$\gamma^d_2$'])

plt.figure(dpi=300, figsize=(12,6))
sns.kdeplot(df, fill=True, alpha=0.5, common_norm=True)
plt.title(fr'Joint probability distribution $\rho(m|d,e)$', fontsize=16, fontweight='bold')
plt.xlabel('Parameter value')
plt.xlim(0,1)
plt.savefig(f'{savedir}/seir_25p.png')
plt.show()

print(np.mean(errors), np.std(errors))
logging.info(f'{np.mean(errors)}, {np.std(errors)}')
