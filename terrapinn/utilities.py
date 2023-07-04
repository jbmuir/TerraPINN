import jax
import jax.numpy as jnp

def pytree_size(pytree):
    """Count the total number of parameters in a JAX PyTree"""
    return jax.tree_util.tree_reduce(lambda x, y: x+y, jax.tree_map(lambda x:x.size, pytree), 0)

def gaussian(x, mu, sd, a):
    return a*jnp.exp(-0.5*jnp.sum(jnp.square((x-mu)/sd)))

def gaussian_c(x, c0=1.0):
        c = c0 + gaussian(x, jnp.array([0.3,0.3]), jnp.array([0.3,0.3]), -0.7)
        c += gaussian(x, jnp.array([-0.5,-0.5]), jnp.array([0.4,0.4]), -0.6)
        c += gaussian(x, jnp.array([-0.2,0.2]), jnp.array([0.2,0.2]), 0.7)
        c += gaussian(x, jnp.array([0.3,-0.4]), jnp.array([0.3,0.3]), 0.6)
        return c
    
jvgaussian_c = jax.jit(jax.vmap(gaussian_c))