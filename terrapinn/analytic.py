import jax.numpy as jnp

# def gauss_i(t, T, shift): 

def gauss(t, T, shift):
    return jnp.exp(-jnp.square((t-shift))/(2*T**2))/(T * jnp.sqrt(2*jnp.pi))

def gauss_d(t, T, shift): 
    return (shift-t)/(T**3 * jnp.sqrt(2*jnp.pi)) * jnp.exp(-jnp.square((t-shift)) / (2*T**2))

def explosion_analytic_2d(t, r, m, md, c=1.0, rho=1.0):
    pass

def explosion_analytic_3d(t, r, m, md, c=1.0, rho=1.0):
    trc = t-r/c
    return (m(trc)/r**2+md(trc)/r)/(4*jnp.pi*rho*c**2)

# def gauss_dd(t, T, shift):

