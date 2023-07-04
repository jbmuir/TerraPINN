import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.tree_util import Partial
# from typing import Callable, Iterable, Optional
from tqdm import tqdm

class HeterogeneousXYModel():

    def __init__(self, axi_model,
                       axi_params,
                       model = None,
                       scheduler = None,
                       optimizer = None,
                       dset_rng_key = jax.random.PRNGKey(43771120),
                       train_rng_key = jax.random.PRNGKey(94899109),
                       hbatch_size = 10000,
                       epoch_iters = 300,
                       epochs = 50, 
                       max_rad = 1.0,
                       amp_scale=jnp.log(2), 
                       tau_scale=jnp.log(2)):
        
        self.axi_model = axi_model
        self.axi_params = axi_params
        self.dset_rng_key = dset_rng_key
        self.train_rng_key = train_rng_key
        self.hbatch_size = hbatch_size
        self.epoch_iters = epoch_iters
        self.epochs = epochs
        self.max_rad = max_rad
        self.amp_scale = amp_scale
        self.tau_scale = tau_scale

        if model is not None:
            self.model = model
        else:
            self.model = self.default_model()
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = optax.warmup_cosine_decay_schedule(
                                                                init_value=1e-3,
                                                                peak_value=1e-2,
                                                                warmup_steps=5000,
                                                                decay_steps=25_000,
                                                                end_value=1e-8,
                                                                )
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optax.adabelief(learning_rate=self.scheduler)

    def default_model(self):
        def amp_tau_model_fn(txy):
            mlp = hk.Sequential([
            hk.Linear(64), jax.nn.swish,
            hk.Linear(32), jax.nn.swish,
            hk.Linear(32), jax.nn.swish,
            hk.Linear(16), jax.nn.swish,
            hk.Linear(4, with_bias=False), # output all a, t, r components 
            ])
            return mlp(txy)
    
        return hk.without_apply_rng(hk.transform(amp_tau_model_fn))
    
    def model_eval(self, params, t, x, y, t1, sd):
        r = jnp.sqrt(jnp.square(x)+jnp.square(y)) # need to recalculate r for autodiff
        tr = jnp.hstack((t,r))
        txy = jnp.hstack((t,x,y))
        model_out = self.model.apply(params, txy)
        amp_shift = jnp.exp(self.amp_scale*jax.nn.tanh(model_out[0])) # try to ban the zero-solution
        dtr = model_out[1:3]
        dk = jnp.exp(model_out[3])
        coord_shift = tr+dtr
        coord_shift.at[1].set(dk*coord_shift[1])
        coord_shift_clamp = jax.lax.clamp(0.0, coord_shift, self.max_rad)
        se = jax.nn.sigmoid(5*(2-t/t1))*jnp.exp(-0.5*jnp.square(r/sd))
        t2 = jax.nn.tanh(jax.nn.tanh(t/t1))
        return se+t2*amp_shift*self.axi_model.apply(self.axi_params, coord_shift_clamp)

    def full_loss_fn(self, params, t, x, y, u, weights=1):
        fep = Partial(self.model_eval, params)
        upred = jax.vmap(fep)(t, x, y)
        loss = jnp.mean(weights*jnp.square((upred-u)))/2
        return loss

    def full_phys_loss_fn(self, params, t, x, y, c, weights=1):
        fep = Partial(self.model_eval, params)
        c2 = c**2
        upred_tt = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=0), argnums=0))(t,x,y).reshape(-1,1)
        upred_xx = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=1), argnums=1))(t,x,y).reshape(-1,1)
        upred_yy = jax.vmap(jax.jacrev(jax.jacrev(fep, argnums=2), argnums=2))(t,x,y).reshape(-1,1)
        res = upred_tt - c2*upred_xx - c2*upred_yy
        loss = jnp.mean(weights*jnp.square(res))/2
        return (loss, res)