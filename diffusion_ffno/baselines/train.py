import jax.numpy as jnp
import equinox as eqx

from jax.tree_util import tree_map
from jax import random, vmap

def compute_loss(model, input, target, x):
    output = vmap(model, in_axes=(0, None))(input, x)
    l = jnp.mean(jnp.linalg.norm((output - target).reshape(input.shape[0], -1), axis=1))
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss


def make_step_scan_with_val(carry, n, optim):
    model, features, targets, x, opt_state, ind_val = carry
    loss, grads = compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    loss_val = compute_loss(model, features[ind_val], targets[ind_val], x)
    return [model, features, targets, x, opt_state, ind_val], [loss, loss_val]