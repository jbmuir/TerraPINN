import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax

def train_to_data(params, model_eval, optimizer, loss_fn, data_iterator, num_batches=30000):
    opt_state = optimizer.init(params)
    @jax.jit
    def step(params, opt_state, *data):
        # all of our loss functions also give the residual at each element of the batch, so need to use has_aux=True
        (loss_value, sres), grads = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(model_eval, params, *data)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, sres
    
    with tqdm(enumerate(data_iterator), desc="Epoch", total=num_batches) as epochiter:
        for i, data in epochiter:
            params, opt_state, loss_value, sres = step(params, opt_state, *data)
            if i%100 == 0:
                epochiter.set_postfix(loss=loss_value)

    return params


def train_to_physics(rng_key, params, model_eval, optimizer, loss_fn, data_sampler, c2_eval, 
                     ndims=3,
                     init_coords_and_weights=None,
                     hbatch_size = 10000,
                     epoch_iters = 300,
                     epochs = 100,
                     anneal_schedule = jnp.ones(100), 
                     bandwidth_schedule = 0.2*jnp.ones(100)):
    
    assert epochs == len(anneal_schedule)
    epoch_loss_history = []

    if init_coords_and_weights != None:
        coords, colloc_weights = init_coords_and_weights
    else:
        coords, colloc_weights = data_sampler(rng_key, ndims, hbatch_size=hbatch_size, bandwidth=bandwidth_schedule[0])

    c2 = c2_eval(jnp.hstack(coords[1:]), anneal_schedule[0]).reshape(-1,1)
    
    @jax.jit
    def step(params, opt_state, c2, *coords, colloc_weights=1): #need to add colloc_weights for importance sampling correction
        (loss_value, sres), grads = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(model_eval, params, c2, *coords, weights=colloc_weights)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, sres
    
    with tqdm(range(1,epochs+1), desc="Epoch") as epochiter:
        for i in epochiter:
            opt_state = optimizer.init(params)
            batch_loss_history = []
            with tqdm(enumerate(range(epoch_iters)), desc="Iter", total=epoch_iters, leave=False) as batchiter:
                for j in batchiter:
                    params, opt_state, loss_value, sres = step(params, opt_state, c2, *coords, colloc_weights=colloc_weights) # original version has collocation point weights and perturbation scaling bounds set here, may need to put it back in...
                    batchiter.set_postfix(loss=loss_value)
                    batch_loss_history += [loss_value]

            bml = jnp.mean(jnp.array(batch_loss_history))
            epoch_loss_history += [bml]
            epochiter.set_postfix(mean_loss=bml) 
            coords, colloc_weights = data_sampler(rng_key, ndims, weights=jnp.sqrt(sres.flatten()), old_coords=coords, hbatch_size=hbatch_size, bandwidth=float(bandwidth_schedule[i]))
            c2 = c2_eval(jnp.hstack(coords[1:]), anneal_schedule[min(i,epochs-1)]).reshape(-1,1)

    
    return (params, epoch_loss_history, coords, colloc_weights) # note that return coords & colloc_weights will be for the *next* epoch (i.e. they have not been trained on; this is so that you can chain runs of train_to_physics if desired)