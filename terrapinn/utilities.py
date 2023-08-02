import jax
import jax.numpy as jnp

def pytree_size(pytree):
    """Count the total number of parameters in a JAX PyTree"""
    return jax.tree_util.tree_reduce(lambda x, y: x+y, jax.tree_map(lambda x:x.size, pytree), 0)

def scale_param_layer(params, scale, layer):
    for key in params.keys():
        if key == f"linear_{layer}":
            for key2 in params[key].keys():
                params[key][key2] = scale*params[key][key2]
    
    return params

def scale_param(params, scale):
    for key in params.keys():
        for key2 in params[key].keys():
            params[key][key2] = scale*params[key][key2]
    
    return params


def gaussian(x, mu, sd, a):
    return a*jnp.exp(-0.5*jnp.sum(jnp.square((x-mu)/sd)))

def gaussian_c(x, a, c0=1.0):
        c = c0 + a*gaussian(x, jnp.array([0.3,0.3]), jnp.array([0.3,0.3]), -0.7)
        c += a*gaussian(x, jnp.array([-0.5,-0.5]), jnp.array([0.4,0.4]), -0.6)
        c += a*gaussian(x, jnp.array([-0.2,0.2]), jnp.array([0.2,0.2]), 0.7)
        c += a*gaussian(x, jnp.array([0.3,-0.4]), jnp.array([0.3,0.3]), 0.6)
        return c
    
jvgaussian_c = jax.jit(jax.vmap(gaussian_c, in_axes=(0,None)))
