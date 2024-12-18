import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='Train the flax pde model')
parser.add_argument('-c', '--config', type=str, help='Path to configuration file', required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['cuda'])

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
import wandb

from solver import pde_solution
from kl_extension import KLExpansion
from utils import get_d_from_u


wandb.init(
    # set the wandb project where this run will be logged
    project="inv_pr_fm_pde",

    # track hyperparameters and run metadata
    config=config
)

savedir = config["savedir"]
datadir = "data/point3_100k"

points = config["points"]

os.makedirs(savedir, exist_ok=True)

import logging
logging.basicConfig(filename=f'{savedir}/logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(jax.default_backend())

# 1. Generation
SIZE = config["dataset_size"]
m = jnp.load(f'{datadir}/m.npy')[:SIZE]
d = jnp.load(f'{datadir}/{config["d_name"]}.npy')[:SIZE]
e = jnp.load(f'{datadir}/e.npy')[:SIZE]

# 2. Train
class MLP(nn.Module):
    out_dim: int = 1
    w: int = config["w"]

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

shapes = {'m': m.shape[1],
         'e': e.shape[1],
         'd': d.shape[1]}
dim_shape = shapes['m'] + shapes['e'] + shapes['d']

model = MLP(out_dim=shapes['m'])

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
def loss_ffm_function(params, x1, x0, d, e, key):
    key, subkey = jax.random.split(key)
    t = jax.random.uniform(subkey, (x0.shape[0],))
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
batch_size = config["batch_size"]
num_epochs = config["epochs"]
learning_rate = config["learning_rate"]
optimizer = optax.adamw(learning_rate=learning_rate)
params = model.init(key, jnp.ones((1, dim_shape+1)))
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params["params"],
    tx=optimizer
)

dataset = jnp.concatenate([m, e, d], axis = 1)

def get_batches(dataset, batch_size, key):
    num_samples = dataset.shape[0]
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield dataset[batch_indices]

losses = [np.inf]
start = time.time()

for epoch in tqdm(range(num_epochs)):
    key, subkey = jax.random.split(key)
    losses_per_epoch = []

    for batch in get_batches(dataset, batch_size, subkey):
        x1 = batch[:, :shapes['m']]
        d = batch[:, (shapes['m']+shapes['e']):]
        e = batch[:, shapes['m']:(shapes['m']+shapes['e'])]

        subkey, batch_key = jax.random.split(subkey)
        x0 = jax.random.uniform(batch_key, (batch.shape[0], shapes['m'])) 

        loss, grads = jax.value_and_grad(loss_ffm_function, has_aux=False)(state.params, x1, x0, d, e, batch_key)
        state = update_model(state, grads)
        losses_per_epoch.append(loss.item())
        wandb.log({"loss_in_epoch": loss.item()})

    epoch_avg_loss = np.mean(losses_per_epoch)
        

    if epoch_avg_loss < np.min(losses):
        with open(f'{savedir}/w_best.pkl', 'wb') as f:
            pickle.dump(state.params, f)

    losses.append(epoch_avg_loss)
    wandb.log({"loss": epoch_avg_loss})

    if (epoch + 1) % 1000 == 0:
        end = time.time()
        logging.info(f"{epoch + 1}: loss {epoch_avg_loss:0.3f} time {(end - start):0.2f}")
        start = end

# 3. Saving
np.save(f"{savedir}/losses.npy", np.array(losses))

with open(f'{savedir}/w.pkl', 'wb') as f:
    pickle.dump(state.params, f)

# 4. Inference

def ode_function(t, m, d, e):
    m = m.reshape(1, -1)
    t = jnp.array(t).reshape(1,-1)
    inputs = jnp.concatenate([m, d, e, t], axis=1)
    return predict(state.params, inputs)[0]

kl = KLExpansion(grid=(64, 64))
kl.calculate_eigh()

m = np.random.normal(size = shapes['m'])
log_kappa = kl.expansion(m)

print(m)
logging.info(f'm = {list(m)}')

errors = []
sols = []

for i in tqdm(range(config["inference_num"])):
    subkey, batch_key = jax.random.split(subkey)
    m0 = jax.random.uniform(batch_key, (1, shapes['m'])) 
    e = np.array([0.2, 0.9])
    u = pde_solution(log_kappa, (e[0], e[1]), verbose=False)
    d = get_d_from_u(u, points)
    m = jnp.array(m).reshape(1,-1)
    e = jnp.array(e).reshape(1,-1)
    d = jnp.array(d).reshape(1,-1)
    solution = solve_ivp(ode_function, t_span=[0, 1], y0=m0[0], t_eval=None, args=(d, e))
    log_kappa = kl.expansion(solution.y[:, -1])
    u = pde_solution(log_kappa, e[0], verbose=False)
    d_pred = get_d_from_u(u, points)
    sols.append(solution.y[:, -1])
    try:
        error = np.linalg.norm(d - d_pred) / np.linalg.norm(d)
        errors.append(error)
        wandb.log({'inference_error': error})
    except ValueError:
        pass


print(np.mean(errors), np.std(errors))
logging.info(f'{np.mean(errors)}, {np.std(errors)}')

wandb.summary['mean_d_err_1k'] = np.mean(errors)
wandb.summary['std_d_err_1k'] = np.std(errors)

wandb.finish()

np.save(f'{savedir}/m.npy', m)
np.save(f'{savedir}/m_avg.npy', np.mean(sols, axis=0))
np.save(f'{savedir}/sols.npy', np.array(sols))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

df = pd.DataFrame(sols, columns = [f'param_{i}' for i in range(shapes['m'])])

plt.figure(dpi=300, figsize=(12,6))
sns.kdeplot(df, fill=True, alpha=0.5, common_norm=True)
plt.title(fr'Joint probability distribution $\rho(m|d,e)$', fontsize=16, fontweight='bold')
plt.xlabel('Parameter value')
# plt.xlim(0,1)
plt.savefig(f'{savedir}/pde_params.png')
plt.show()

log_kappa = kl.expansion(m[0])
u = pde_solution(log_kappa, e[0], verbose=False)
log_kappa_pred = kl.expansion(np.mean(sols, axis=0))
u_pred = pde_solution(log_kappa_pred, e[0], verbose=False)

plt.figure(figsize=(12, 8), dpi=200)
plt.subplot(221)
plt.contourf(kl.X_mesh*64, kl.Y_mesh*64, log_kappa.T, levels=50, cmap='viridis')
plt.colorbar()
plt.title(fr'$\log \kappa$ true')
plt.subplot(222)
plt.contourf(kl.X_mesh*64, kl.Y_mesh*64, u.T, levels=50, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.scatter([x[1] for x in points],[x[0] for x in points], color='r', marker='*')
plt.title(fr'Solution of $-\nabla \cdot (\kappa \nabla u) = 0$')
plt.subplot(223)
plt.contourf(kl.X_mesh*64, kl.Y_mesh*64, log_kappa_pred.T, levels=50, cmap='viridis')
plt.colorbar()
plt.title(fr'$\log \kappa$ pred')
plt.subplot(224)
plt.contourf(kl.X_mesh*64, kl.Y_mesh*64, u_pred.T, levels=50, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.scatter([x[1] for x in points],[x[0] for x in points], color='r', marker='*')
plt.title(fr'Solution of $-\nabla \cdot (\kappa \nabla u) = 0$')
plt.savefig(f'{savedir}/avg_pred_t.png')
plt.show()
