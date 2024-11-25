import jax.numpy as jnp
from jax import vmap, random
from jax.lax import scan

def get_predictions_scan(carry, ind, N_samples, integrator_, prior):
    model, targets, features, x, t, dt, key = carry
    key_, key = random.split(key, 2)
    targets_0 = vmap(prior)(random.split(key_, N_samples))
    predictions = scan(integrator_, [model, targets_0, features[ind], x, dt], t)[0][1]
    return [model, targets, features, x, t, dt, key], predictions

def get_statistics_scan(carry, ind, N_samples, integrator_, prior):
    model, targets, features, x, t, dt, key = carry
    key_, key = random.split(key, 2)
    targets_0 = vmap(prior)(random.split(key_, N_samples))
    predictions = scan(integrator_, [model, targets_0, features[ind], x, dt], t)[0][1]
    predictions_mean = jnp.mean(predictions, axis=0)
    predictions_var = jnp.sqrt(jnp.var(predictions, axis=0))
    return [model, targets, features, x, t, dt, key], jnp.stack([predictions_mean, predictions_var], axis=0)

def compute_error(targets, predictions):
    errors = jnp.linalg.norm((predictions - targets).reshape(targets.shape[0], targets.shape[1], -1), axis=2) / jnp.linalg.norm((targets).reshape(targets.shape[0], targets.shape[1], -1), axis=2)
    return errors