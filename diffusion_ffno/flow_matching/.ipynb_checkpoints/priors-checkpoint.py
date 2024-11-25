import jax.numpy as jnp
import itertools

from jax import random

def get_basis_normal_periodic(x, N):
    inds = itertools.product(range(N), repeat=x.shape[0])
    b = []
    f = []
    for ind in list(inds)[:N]:
        freq = 2*jnp.pi*jnp.array(ind).reshape([-1,] + [1,]*(x.shape[0]))
        args = jnp.sum(1j*x*freq, axis=0)
        b.append(jnp.exp(args))
        f.append(jnp.sum(freq**2))
    b = jnp.stack(b, 0)
    f = jnp.stack(f, 0)
    return b, f.reshape([-1,] + [1,]*(b.ndim-1))

def normal_periodic(key, params):
    b, f, scale, po = params
    s = [b.shape[0],] + [1,]*(b.ndim-1)
    c = random.normal(key, shape=s, dtype=jnp.complex64)
    return jnp.real(jnp.sum(b * c / (1 + scale*f)**po, axis=0, keepdims=True))