from jax import vmap

def explicit_Euler(model, v_t, u, x, t, dt):
    return v_t + model(v_t, u, x, t)*dt

def integrator(carry, t, method):
    model, v_0, features, x, dt = carry
    v_0 = vmap(method, in_axes=(None, 0, None, None, None, None))(model, v_0, features, x, t, dt)
    return [model, v_0, features, x, dt], None