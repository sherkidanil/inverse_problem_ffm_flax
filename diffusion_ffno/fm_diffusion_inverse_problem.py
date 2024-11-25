import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np

SIZE = 50_000
Diffusion_features = np.load('data/try1/features_7p.npy')[:SIZE]
Diffusion_targets = np.load('data/try1/targets_7p.npy')[:SIZE]

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

from flow_matching import train as train_fm, priors, flows
from flow_matching.architectures import ffno as ffno_fm

from flow_matching.integrators import explicit_Euler, integrator
from flow_matching.evaluate import get_statistics_scan, compute_error as compute_error_fm

x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

Diffusion_features = jnp.array(Diffusion_features)
Diffusion_targets = jnp.array(Diffusion_targets)
Diffusion_coordinates = jnp.stack([X,Y])

# D = 2
# learning_rate = 1e-4
# N_processor = 32
# N_train = int(SIZE*.7)
# N_run = 10_000
# N_batch = 512
# N_layers = 4
# N_modes = 16
# N_drop = N_run // 2
# gamma = 0.5
# scale = 0.001
# po = 2.0
# N = 150

D = 2
learning_rate = 1e-4
N_processor = 32
N_train = int(SIZE*.7)
N_run = 100_000
N_batch = 128
N_layers = 4
N_modes = 16
N_drop = N_run // 2
gamma = 0.5
scale = 0.01
po = 2.0
N = 3


key = random.PRNGKey(11)
keys = random.split(key, 3)
N_features = [Diffusion_coordinates.shape[0] + Diffusion_features.shape[1] + Diffusion_targets.shape[1] + 1, N_processor, Diffusion_targets.shape[1]]
model = ffno_fm.flow_FFNO(N_layers, N_features, N_modes, D, keys[0])

learning_rate = optax.exponential_decay(learning_rate, N_drop, gamma)
optim = optax.adamw(learning_rate=learning_rate)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

ind = jnp.arange(Diffusion_features.shape[0])
ind_train, ind_test = ind[:N_train], ind[N_train:]
n = random.choice(keys[1], ind_train, shape = (N_run, N_batch))

carry = [model, Diffusion_targets, Diffusion_features, Diffusion_coordinates, opt_state, keys[2]]

flow_params = [0.0, ]
flow = lambda target_1, target_0, t: flows.optimal_transport(target_1, target_0, t, flow_params)

basis, freq = priors.get_basis_normal_periodic(Diffusion_coordinates, N)
prior_params = [basis, freq, scale, po]
prior = lambda key: priors.normal_periodic(key, prior_params)
some_noise = prior(key)
print(key.shape)
prior = lambda key: random.normal(key, shape=some_noise.shape)

make_step_scan_ = lambda a, b: train_fm.make_step_scan(a, b, optim, flow, prior)
carry, losses = scan(make_step_scan_, carry, n)
model = carry[0]

plt.figure(dpi=300)
plt.yscale("log")
plt.plot(losses);
plt.savefig("imgs/loss_2.png")

N_t = 50
N_samples = 50
t = jnp.linspace(0, 1, N_t)
dt = t[1] - t[0]
integrator_ = lambda carry, t: integrator(carry, t, explicit_Euler)

get_statistics_scan_ = lambda carry, ind: get_statistics_scan(carry, ind, N_samples, integrator_, prior)
_, train_predictions = scan(get_statistics_scan_, [model, Diffusion_targets, Diffusion_features, Diffusion_coordinates, t, dt, keys[3]], ind_train[:10])
_, test_predictions = scan(get_statistics_scan_, [model, Diffusion_targets, Diffusion_features, Diffusion_coordinates, t, dt, keys[3]], ind_test[:10])

train_errors = compute_error_fm(Diffusion_targets[ind_train[:10]], train_predictions[:, 0])
test_errors = compute_error_fm(Diffusion_targets[ind_test[:10]], test_predictions[:, 0])

mean_train_error = jnp.mean(train_errors, axis=0)
mean_test_error = jnp.mean(test_errors, axis=0)

print("train errors", mean_train_error)
print("test errors", mean_test_error);

train_correlations = vmap(jnp.corrcoef, in_axes=(0, 0))(jnp.abs(train_predictions[:, 0, 0] - Diffusion_targets[ind_train[:10]][:, 0]).reshape(10, -1), train_predictions[:, 1, 0].reshape(10, -1))[:, 0, 1]
test_correlations = vmap(jnp.corrcoef, in_axes=(0, 0))(jnp.abs(test_predictions[:, 0, 0] - Diffusion_targets[ind_test[:10]][:, 0]).reshape(10, -1), test_predictions[:, 1, 0].reshape(10, -1))[:, 0, 1]

mean_train_correlation = jnp.mean(train_correlations, axis=0)
mean_test_correlation = jnp.mean(test_correlations, axis=0)

print("train correlation", mean_train_correlation)
print("test correlation", mean_test_correlation);

plt.figure(dpi=150, figsize = (18, 6))
plt.subplot(131)
plt.contourf(X, Y, train_predictions[ind_train[0], 0, 0])
plt.title('prediction')
plt.subplot(132)
plt.contourf(X, Y, Diffusion_targets[ind_train[0], 0])
plt.title('target')
plt.subplot(133)
plt.contourf(X, Y, Diffusion_features[ind_train[0], 0])
plt.title('feature')
plt.savefig('imgs/fm_res.png')