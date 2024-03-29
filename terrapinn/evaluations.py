import jax
import jax.numpy as jnp

TANHONE = float(jnp.tanh(1))

def radial_model_eval(model, params, t, r, t1=1.0, sd=0.01):
    tr = jnp.hstack((t,r))
    nn = model.apply(params, tr)
    se = jax.nn.sigmoid(5*(2-t/t1))*jnp.exp(-0.5*jnp.square(r/sd))
    t2 = jax.nn.tanh(jax.nn.tanh(t/t1))
    return se+t2*nn

def model_eval_2d(axi_model, axi_params, model, params, t, x, y, t1=1.0, sd=0.01, amp_scale=jnp.log(2), max_rad=1.0):
    r = jnp.sqrt(jnp.square(x)+jnp.square(y)) # need to recalculate r for autodiff
    tr = jnp.hstack((t,r))
    txy = jnp.hstack((t,x,y))
    model_out = model.apply(params, txy)
    amp_shift = jnp.exp(amp_scale*jax.nn.tanh(model_out[0])) # try to ban the zero-solution
    dtr = model_out[1:3]
    dk = jnp.exp(model_out[3])
    coord_shift = tr+dtr
    coord_shift.at[1].set(dk*coord_shift[1])
    coord_shift_clamp = jax.lax.clamp(0.0, coord_shift, max_rad)
    se = jax.nn.sigmoid(5*(2-t/t1))*jnp.exp(-0.5*jnp.square(r/sd))
    t2 = jax.nn.tanh(jax.nn.tanh(t/t1))/TANHONE
    return se+t2*amp_shift*axi_model.apply(axi_params, coord_shift_clamp)

# def model_eval_2d(axi_model, axi_params, model, params, t, x, y, t1=1.0, sd=0.01, amp_scale=jnp.log(2), max_rad=1.0):
#     r = jnp.sqrt(jnp.square(x)+jnp.square(y)) # need to recalculate r for autodiff
#     tr = jnp.hstack((t,r))
#     txy = jnp.hstack((t,x,y))
#     model_out = model.apply(params, txy)
#     amp_shift = jnp.exp(amp_scale*jax.nn.tanh(model_out[0])) # try to ban the zero-solution
#     dtr = model_out[1:3]
#     dk = jnp.exp(model_out[3])
#     coord_shift = tr+dtr
#     coord_shift.at[1].set(dk*coord_shift[1])
#     coord_shift_clamp = jax.lax.clamp(0.0, coord_shift, max_rad)
#     ref = axi_model.apply(axi_params,tr)
#     t2 = jax.nn.tanh(jax.nn.tanh(t/t1))/TANHONE
#     return (1-t2)*ref+t2*amp_shift*axi_model.apply(axi_params, coord_shift_clamp)