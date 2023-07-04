import jax
import jax.numpy as jnp
from jax.tree_util import Partial

def u_loss_fn_1d(model_eval, params, t, r, u, weights=1):
    pmodel = Partial(model_eval, params)
    upred = jax.vmap(pmodel)(t,r)
    res = upred-u
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def u_loss_fn_2d(model_eval, params, t, x, y, u, weights=1):
    fep = Partial(model_eval, params)
    upred = jax.vmap(fep)(t,x,y)
    res = upred-u
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def u_loss_fn_3d(model_eval, params, t, x, y, z, u, weights=1):
    fep = Partial(model_eval, params)
    upred = jax.vmap(fep)(t,x,y,z)
    res = upred-u
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def phys_loss_fn_2d(model_eval, params, t, x, y, c2, weights=1):
    fep = Partial(model_eval, params)
    upred_tt = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=0), argnums=0))(t,x,y).reshape(-1,1)
    upred_xx = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=1), argnums=1))(t,x,y).reshape(-1,1)
    upred_yy = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=2), argnums=2))(t,x,y).reshape(-1,1)
    res = upred_tt - c2*upred_xx - c2*upred_yy
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def phys_loss_fn_3d(model_eval, params, t, x, y, z, c2, weights=1):
    fep = Partial(model_eval, params)
    upred_tt = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=0), argnums=0))(t,x,y,z).reshape(-1,1)
    upred_xx = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=1), argnums=1))(t,x,y,z).reshape(-1,1)
    upred_yy = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=2), argnums=2))(t,x,y,z).reshape(-1,1)
    upred_zz = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=3), argnums=3))(t,x,y,z).reshape(-1,1)
    res = upred_tt - c2*upred_xx - c2*upred_yy - c2*upred_zz
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def phys_loss_fn_2x2d(model_eval, params, t, xr, yr, xs, ys, c2, weights=1):
    fep = Partial(model_eval, params)
    upred_tt = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=0), argnums=0))(t,xr,yr,xs,ys).reshape(-1,1)
    upred_xx = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=1), argnums=1))(t,xr,yr,xs,ys).reshape(-1,1)
    upred_yy = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=2), argnums=2))(t,xr,yr,xs,ys).reshape(-1,1)
    res = upred_tt - c2*upred_xx - c2*upred_yy
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)

def phys_loss_fn_3x3d(model_eval, params, t, xr, yr, zr, xs, ys, zs, c2, weights=1):
    fep = Partial(model_eval, params)
    upred_tt = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=0), argnums=0))(t,xr,yr,zr,xs,ys,zs).reshape(-1,1)
    upred_xx = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=1), argnums=1))(t,xr,yr,zr,xs,ys,zs).reshape(-1,1)
    upred_yy = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=2), argnums=2))(t,xr,yr,zr,xs,ys,zs).reshape(-1,1)
    upred_zz = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=3), argnums=3))(t,xr,yr,zr,xs,ys,zs).reshape(-1,1)
    res = upred_tt - c2*upred_xx - c2*upred_yy - c2*upred_zz
    loss = jnp.mean(weights*jnp.square(res))/2
    return (loss, res)