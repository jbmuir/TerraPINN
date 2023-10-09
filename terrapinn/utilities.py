import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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

def plot_1d_slice():
    pass

def plot_2d_slice(eval_fn, slice_axis, slice_value, trange=(0,1), xrange=(-1,1), tpoints=51, xpoints=100):
    if slice_axis == 0:
        r1 = jnp.linspace(xrange[0], xrange[1], xpoints)
    else:
        r1 = jnp.linspace(trange[0], trange[1], tpoints)
    r2 = jnp.linspace(xrange[0], xrange[1], xpoints)
    R1, R2 = jnp.meshgrid(r1,r2)
    pdims = R1.shape
    R1 = R1.flatten().reshape(-1,1)
    R2 = R2.flatten().reshape(-1,1)
    S = slice_value * jnp.ones((len(R1),1))
    if slice_axis == 0:
        coords = (S,R1,R2)
    elif slice_axis == 1:
        coords = (R1,S,R2)
    elif slice_axis == 2:
        coords = (R1,R2,S)

    output = jax.vmap(eval_fn)(*coords)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contourf(r1, r2, jnp.reshape(output,pdims))
    fig.show()
    return (coords, jnp.reshape(output,pdims), fig)

    
