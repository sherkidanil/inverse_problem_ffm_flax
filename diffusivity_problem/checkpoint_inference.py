import os
import argparse
import yaml
    
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

import matplotlib.pyplot as plt

savedir = 'models/point5_try2_2kk'

with open(f'{savedir}/w_best.pkl', 'rb') as f:
    params = pickle.load(f)

class MLP(nn.Module):
    out_dim: int = 1
    w: int = 128

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

points = [(16, 16),
          (16, 48),
          (32, 32),
          (48, 16),
          (48, 48)
         ]

shapes = {'m': 16,
         'e': 2,
         'd': len(points)}
dim_shape = shapes['m'] + shapes['e'] + shapes['d']

model = MLP(out_dim=shapes['m'])

@jax.jit
def predict(params, inputs):
    return model.apply({"params": params}, inputs)


subkey = jax.random.PRNGKey(0)

def ode_function(t, m, d, e):
    m = m.reshape(1, -1)
    t = jnp.array(t).reshape(1,-1)
    inputs = jnp.concatenate([m, d, e, t], axis=1)
    return predict(params, inputs)[0]

kl = KLExpansion(grid=(64, 64))
kl.calculate_eigh()

m = np.random.normal(size = shapes['m'])
log_kappa = kl.expansion(m)

print(m)

errors = []
errors_u = []
sols = []

for i in tqdm(range(100)):
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
    u_pred = pde_solution(log_kappa, e[0], verbose=False)
    d_pred = get_d_from_u(u_pred, points)
    sols.append(solution.y[:, -1])
    try:
        error = np.linalg.norm(d - d_pred) / np.linalg.norm(d)
        errors.append(error)
        error = np.linalg.norm(u - u_pred)/np.linalg.norm(u)
        errors_u.append(error)
    except ValueError:
        pass


print(np.mean(errors), np.std(errors))
print(np.mean(errors_u), np.std(errors_u))

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
plt.savefig(f'{savedir}/avg_pred_t_from_checkpoint.png')
plt.show()