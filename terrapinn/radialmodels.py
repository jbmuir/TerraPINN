import haiku as hk
import jax
import jax.numpy as jnp
import optax
# from jax.tree_util import Partial
# from typing import Callable, Iterable, Optional
from tqdm import tqdm

class HomogenousDataModelXY():
    """Class for generating and training a homogenous XY model on data (i.e. only radial dependency)"""
    
    def __init__(self, t_data, r_data, u_data, 
                       model = None,
                       scheduler = None,
                       optimizer = None,
                       dset_rng_key = jax.random.PRNGKey(43771120),
                       train_rng_key = jax.random.PRNGKey(94899109),
                       num_batches = 30000,
                       batch_size = 10000,
                       max_radius = 1.0):
        
        self.ndims = 2
        self.t_data = t_data
        self.r_data = r_data
        self.u_data = u_data
        self.dset_rng_key = dset_rng_key
        self.train_rng_key = train_rng_key
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.max_radius = max_radius
        self.dset_iterator = self.make_dset_iterator()
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
                                                                decay_steps=25000,
                                                                end_value=1e-8,
                                                                )
            
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optax.adabelief(learning_rate=self.scheduler)

    
    def default_model(self):
        def radial_model_fn(tr):
            mlp = hk.Sequential([
              hk.Linear(32), jax.nn.swish,
              hk.Linear(32), jax.nn.swish,
              hk.Linear(1),
            ])
            return mlp(tr)

        model = hk.without_apply_rng(hk.transform(radial_model_fn))
        return model
    
    def make_radial_dataset_iterator(self):
        split_rng_key = jax.random.split(self.dset_rng_key, self.num_batches)
        tvv = self.t_data.reshape(-1,1)
        rvv = self.r_data.reshape(-1,1)
        uvv = self.u_data.reshape(-1,1)
        def _dset_generator():
            for i in range(self.num_batches):
                idx = jax.random.choice(split_rng_key[i], len(tvv), (self.batch_size,))
                yield (tvv[idx], rvv[idx], uvv[idx])
        
        return _dset_generator()
    
    def model_eval(self, params, t, r, t1, sd):
        tr = jnp.hstack((t,r))
        nn = self.model.apply(params, tr)
        se = jax.nn.sigmoid(5*(2-t/t1))*jnp.exp(-0.5*jnp.square(r/sd))
        t2 = jax.nn.tanh(jax.nn.tanh(t/t1))
        return se+t2*nn

    def loss_fn(self, params, t, r, u):
        upred = self.model_eval(params, t, r)
        loss = jnp.mean(jnp.square((upred-u)))/2
        return loss
    
    def train(self, params = None, opt_state = None):
        if params is None:
            dummy_t, dummy_r, _ = next(self.dset_iterator)
            params = self.model.init(self.train_rng_key, jnp.hstack((dummy_t, dummy_r)))
        
        if opt_state is None:
            opt_state = self.optimizer.init(params)

        @jax.jit
        def step(params, opt_state, t, r, u):
            loss_value, grads = jax.value_and_grad(self.loss_fn)(params, t, r, u)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value
        
        with tqdm(enumerate(self.dset_iterator), desc="Epoch", total=self.num_batches) as epoch_iter:
            for i, (t, r, u) in epoch_iter:
                params, opt_state, loss_value = step(params, opt_state, t, r, u)
                if i%100 == 0:
                    epoch_iter.set_postfix(loss=loss_value)

        return (params, opt_state)
