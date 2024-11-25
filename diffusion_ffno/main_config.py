import argparse
import yaml

parser = argparse.ArgumentParser(
                    prog='FM_Diff',
                    description='Flow matching training',
                    epilog='python main.py -c config.yaml')

parser.add_argument('-c', '--config')

args = parser.parse_args()

if args.config is None:
    config = 'configs/base.yaml'
else:
    config = args.config

with open(config, 'r') as f:
    config = yaml.safe_load(f)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['cuda'])

SIZE = config['size']
Diffusion_features = np.load(config['features'])[:SIZE]

Diffusion_targets = np.load(config['targets'])[:SIZE]
OUT_FODLER = config['out_folder']

x_size, y_size = Diffusion_features[0][0].shape
x = np.linspace(0, 1, x_size)
y = np.linspace(0, 1, y_size)
X, Y = np.meshgrid(x, y, indexing='ij')

Diffusion_features = jnp.array(Diffusion_features)
Diffusion_targets = jnp.array(Diffusion_targets)
Diffusion_coordinates = jnp.stack([X,Y])

D = config['D']
learning_rate = config['lr']
N_processor = config['n_proc']
N_train = int(SIZE*config['frac_train'])
N_run = config['N_run']
N_batch = config['N_batch']
N_layers = config['N_layers']
N_modes = config['N_modes']
N_drop = N_run // 2
gamma = config['gamma']
scale = config['scale']
po = config['po']
N = config['N']


y = random.PRNGKey(11)
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
plt.savefig(f"imgs/{OUT_FOLDER}/loss_2.png")

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
config["train errors"] = mean_train_error
config["test errors"] = mean_test_error

train_correlations = vmap(jnp.corrcoef, in_axes=(0, 0))(jnp.abs(train_predictions[:, 0, 0] - Diffusion_targets[ind_train[:10]][:, 0]).reshape(10, -1), train_predictions[:, 1, 0].reshape(10, -1))[:, 0, 1]
test_correlations = vmap(jnp.corrcoef, in_axes=(0, 0))(jnp.abs(test_predictions[:, 0, 0] - Diffusion_targets[ind_test[:10]][:, 0]).reshape(10, -1), test_predictions[:, 1, 0].reshape(10, -1))[:, 0, 1]

mean_train_correlation = jnp.mean(train_correlations, axis=0)
mean_test_correlation = jnp.mean(test_correlations, axis=0)

print("train correlation", mean_train_correlation)
print("test correlation", mean_test_correlation);
config["train correlation"] = mean_train_correlation
config["test correlation"] = mean_test_correlation

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
plt.savefig(f'imgs/{OUT_FOLDER}/fm_res.png')

config['config'] = args.config

with open(f"imgs/{OUT_FOLDER}/{args.config.replace('configs/','')}", 'w'):
    yaml.dump(config)