import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import equinox as eqx
import time
import requests
import itertools

import numpy as np

from urllib.parse import urlencode
from jax import random, vmap
from jax.lax import scan
from jax.tree_util import tree_map

from baselines import evaluate as eval_b, train as train_b
from baselines.architectures import ffno as ffno_b

x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

SIZE = 50_000

Diffusion_features = jnp.load('data/try1/features.npy')[:SIZE]
Diffusion_targets = jnp.load('data/try1/fields_100k.npy')[:SIZE]
Diffusion_coordinates = np.stack([X,Y])


D = 2
learning_rate = 3e-4
N_processor = 32
N_train = 40_000
N_val = 100
N_run = 100
N_batch = 5
N_layers = 4
N_modes = 16
N_drop = N_run // 2
gamma = 0.5

key = random.PRNGKey(11)
keys = random.split(key, 2)
N_features = [Diffusion_coordinates.shape[0] + Diffusion_features.shape[1], N_processor, Diffusion_targets.shape[1]]
model = ffno_b.FFNO(N_layers, N_features, N_modes, D, keys[0])

learning_rate = optax.exponential_decay(learning_rate, N_drop, gamma)
optim = optax.adamw(learning_rate=learning_rate)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

ind = jnp.arange(Diffusion_features.shape[0])
ind_train, ind_val, ind_test = ind[:N_train], ind[N_train:N_train+N_val], ind[N_train+N_val:]
n = random.choice(keys[1], ind_train, shape = (N_run, N_batch))
carry = [model, Diffusion_features, Diffusion_targets, Diffusion_coordinates, opt_state, ind_val]

make_step_scan_ = lambda a, b: train_b.make_step_scan_with_val(a, b, optim)

res, losses = scan(make_step_scan_, carry, n)
model = res[0]

plt.figure(dpi=300)
plt.yscale("log")
plt.plot(losses[0], label='train')
plt.plot(losses[1], label='validation')
plt.legend()
plt.savefig('imgs/loss_ffno_baseline_with_val.png')

_, train_errors = scan(eval_b.compute_error_scan, [model, Diffusion_features, Diffusion_targets, Diffusion_coordinates], ind_train)
_, test_errors = scan(eval_b.compute_error_scan, [model, Diffusion_features, Diffusion_targets, Diffusion_coordinates], ind_test)

mean_train_error = jnp.mean(train_errors, axis=0)
mean_test_error = jnp.mean(test_errors, axis=0)

print("train errors", mean_train_error)
print("test errors", mean_test_error)

