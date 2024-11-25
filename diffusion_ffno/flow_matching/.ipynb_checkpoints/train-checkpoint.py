import jax.numpy as jnp
import equinox as eqx

from jax.tree_util import tree_map
from jax import vmap, random

def compute_loss(model, target_1, target_0, feature, x, t, flow):
    input_, output_ = vmap(flow, in_axes=(0, 0, 0))(target_1, target_0, t)
    output = vmap(model, in_axes=(0, 0, None, 0))(input_, feature, x, t)
    l = jnp.mean(jnp.linalg.norm((output -  output_).reshape(output_.shape[0], -1), axis=1))
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, n, optim, flow, prior):
    model, target, feature, x, opt_state, key = carry
    keys = random.split(key, 3)
    target_0 = vmap(prior)(random.split(keys[0], n.size))
    t = random.uniform(keys[1], (n.size, ))
    loss, grads = compute_loss_and_grads(model, target[n], target_0, feature[n], x, t, flow)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, target, feature, x, opt_state, keys[2]], loss