import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax

def train_to_data(params, model_eval, optimizer, loss_fn, data_iterator, num_batches=30000):
    opt_state = optimizer.init(params)
    @jax.jit
    def step(params, opt_state, *data):
        # all of our loss functions also give the residual at each element of the batch, so need to use has_aux=True
        (loss_value, res), grads = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(model_eval, params, *data)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, res
    
    with tqdm(enumerate(data_iterator), desc="Epoch", total=num_batches) as epochiter:
        for i, data in epochiter:
            params, opt_state, loss_value, res = step(params, opt_state, *data)
            if i%100 == 0:
                epochiter.set_postfix(loss=loss_value)

    return params


# def train_to_physics(params, model_eval, optimizer, loss_fn, data_sampler,  
#                      hbatch_size = 10_000,
#                      epoch_iters = 300,
#                      epochs = 100,
#                      anneal_schedule = jnp.ones(100)):
#     epoch_loss_history = []

#     opt_state = optimizer.init(params)
    
#     x = jax.random.uniform(rm_rng_key, shape=(2*hbatch_size,1), minval=-1, maxval=1)
#     y = jax.random.uniform(rm_rng_key, shape=(2*hbatch_size,1), minval=-1, maxval=1)
#     t = jax.random.uniform(rm_rng_key, shape=(2*hbatch_size,1), minval=0, maxval=1)
#     r = jnp.sqrt(x**2+y**2).reshape(-1,1)
#     c = jnp.array(gaussian_c(jnp.hstack((x,y)), c0)).reshape(-1,1)
#     p = (jnp.ones(len(x))/4).reshape(-1,1)
    
#     @jax.jit
#     def step(rad_params, amp_tau_params, opt_state, t, x, y, c, weights=1, amp_scale=jnp.log(2), tau_scale=jnp.log(2)):
#         (loss_value, res), grads = jax.value_and_grad(full_phys_loss_fn, argnums=1, has_aux=True)(
#                                                                 rad_params, amp_tau_params, t, x, y, c, weights=weights,  amp_scale=amp_scale, tau_scale=tau_scale)
        
#         updates, opt_state = optimizer.update(grads, opt_state, amp_tau_params)
#         amp_tau_params = optax.apply_updates(amp_tau_params, updates)
#         return amp_tau_params, opt_state, loss_value, res
    
#     with tqdm(range(epochs), desc="Epoch") as epochiter:
#         for i in epochiter:
#             opt_state = optimizer.init(amp_tau_params)
#             batch_loss_history = []
#             with tqdm(enumerate(range(epoch_iters)), desc="Iter", total=epoch_iters, leave=False) as batchiter:
#                 for j in batchiter:
#                     amp_tau_params, opt_state, loss_value, res = step(rad_params, amp_tau_params, opt_state, t, x, y, c, weights=1/(4*p), amp_scale=jnp.log(maxampcurve(anneal_schedule[i])),
#                                                                                                                     tau_scale=2*jnp.log(maxampcurve(anneal_schedule[i])))
#                     batchiter.set_postfix(loss=loss_value)
#                     batch_loss_history += [loss_value]

#             bml = jnp.mean(jnp.array(batch_loss_history))
#             epoch_loss_history += [bml]
#             epochiter.set_postfix(mean_loss=bml) 
#             if i < (epochs-1):
#                 #have to convert to float64 to satisfy numpy's checks; also for now a mixture of numpy and jax but should jaxify later
#                 xyt, p = resample(x.flatten(), y.flatten(), t.flatten(), np.array(jnp.square(res.flatten()), dtype=np.float64), hbatch_size)
#                 p = jnp.array(p.reshape(-1,1))
#                 c = jnp.array((anneal_schedule[i+1]*gaussian_c(xyt[:,:2], c0) + (1-anneal_schedule[i+1])*c0).reshape(-1,1))
#                 x = jnp.array(xyt[:,0].reshape(-1,1))
#                 y = jnp.array(xyt[:,1].reshape(-1,1))
#                 t = jnp.array(xyt[:,2].reshape(-1,1))
#                 r = jnp.sqrt(x**2+y**2)

        
#     return amp_tau_params, epoch_loss_history

# optimizer=optax.adabelief(learning_rate=3e-4)
# amp_tau_params, loss_history = full_fit(radial_params, amp_tau_params_anneal, optimizer)