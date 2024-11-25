import jax.numpy as jnp

def compute_error_scan(carry, ind):
    model, features, targets, x = carry
    prediction = model(features[ind], x)
    error = jnp.linalg.norm((prediction - targets[ind]).reshape(targets.shape[1], -1), axis=1) / jnp.linalg.norm((targets[ind]).reshape(targets.shape[1], -1), axis=1)
    return carry, error