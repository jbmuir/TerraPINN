import jax
import jax.numpy as jnp
from jax.tree_util import Partial

def u_loss_fn(model_eval, params, u, *coords, weights=1):
    pmodel = Partial(model_eval, params)
    upred = jax.vmap(pmodel)(*coords)
    res = upred-u
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def phys_loss_fn_2d(model_eval, params, c2, *coords, weights=1):
    fep = Partial(model_eval, params)
    upred_tt = jax.vmap(jax.hessian(fep, argnums=0))(*coords).reshape(-1,1)
    upred_xx = jax.vmap(jax.hessian(fep, argnums=1))(*coords).reshape(-1,1)
    upred_yy = jax.vmap(jax.hessian(fep, argnums=2))(*coords).reshape(-1,1)
    res = upred_tt - c2*upred_xx - c2*upred_yy
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def phys_loss_fn_3d(model_eval, params, c2, *coords, weights=1):
    fep = Partial(model_eval, params)
    upred_tt = jax.vmap(jax.hessian(fep, argnums=0))(*coords).reshape(-1,1)
    upred_xx = jax.vmap(jax.hessian(fep, argnums=1))(*coords).reshape(-1,1)
    upred_yy = jax.vmap(jax.hessian(fep, argnums=2))(*coords).reshape(-1,1)
    upred_zz = jax.vmap(jax.hessian(fep, argnums=3))(*coords).reshape(-1,1)
    res = upred_tt - c2*upred_xx - c2*upred_yy - c2*upred_zz
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)
