import jax
import jax.numpy as jnp
from sklearn.neighbors import KernelDensity
import warnings


def weighted_dataloader(rng_key, *data, weights=None, num_batches=30000, batch_size=10000, jitter=None):
    split_rng_key = jax.random.split(rng_key, num_batches)
    def _dset_generator():
        for i in range(num_batches):
            idx = jax.random.choice(split_rng_key[i], len(data[0]), (batch_size,))
            if jitter is not None:
                jkeys = jax.random.split(split_rng_key[i], len(jitter))
                yield [d[idx]*(1+j*jax.random.normal(jk)) for (d,j,jk) in zip(data, jitter, jkeys)]
            else:
                yield [d[idx] for d in data]
    
    return _dset_generator()

# def kde_resampler(rng_key, coords, weights, n_resample, bandwidth=0.2):
#     kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
#     transformed_coords = jnp.arctanh(coords)
#     transform_jacobian = jnp.prod(1/(1-jnp.square(coords)), axis=1)
#     kde.fit(transformed_coords, sample_weight=weights/transform_jacobian)
#     resampled_coords = kde.sample(n_resample, random_state=int(rng_key[1]))
#     untransformed_resampled_coords = jnp.tanh(resampled_coords)
#     return untransformed_resampled_coords

def uniform_plus_kde_sampler(rng_key, ndims, weights=None, old_coords=None, hbatch_size=10000, bandwidth=0.5, sd=1.0):
    rng_key_1, rng_key_2 = jax.random.split(rng_key)
    sample_half_1 = jax.random.uniform(rng_key_1, (hbatch_size, ndims), minval=-1.0, maxval=1.0)
    if weights == None:
        sample_half_2 = jax.random.uniform(rng_key_2, (hbatch_size, ndims), minval=-1.0, maxval=1.0)
        samples = jnp.vstack((sample_half_1, sample_half_2))
        sample_probability_weight = jnp.ones((2*hbatch_size, 1))
    else:
        kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
        #transform old time coordinate from [sd,1] to [-1,1]
        old_coords = jnp.hstack(old_coords)
        old_coords = old_coords.at[:,0].multiply(2.0+2*sd)
        old_coords = old_coords.at[:,0].add(-1.0-2*sd)
        transformed_coords = jnp.arctanh(old_coords)
        transform_jacobian = jnp.prod(1/(1-jnp.square(old_coords)), axis=1)
        kde.fit(transformed_coords, sample_weight=weights/transform_jacobian) #divide by transform jacobian to get probability in transformed space
        resampled_coords = kde.sample(hbatch_size, random_state=int(rng_key_2[1]))
        sample_half_2 = jnp.tanh(resampled_coords)
        samples = jnp.vstack((sample_half_1, sample_half_2))
        sample_log_probability = kde.score_samples(jnp.arctanh(samples)) - jnp.sum(jnp.log((1-jnp.square(samples))), axis=1) + jnp.log(2+2*sd) # multiply by transform jacobian to get probability in original space, log(2) factor comes from transforming time from [sd,1] to [-1,1]
        #transform time coordinate from [-1,1] to [0,1]     
        uniform_inverse_density = 2*(ndims-1)*(1-sd)
        sample_probability_weight = 1/(0.5+0.5*uniform_inverse_density*jnp.exp(sample_log_probability)) # = uniform / (0.5*uniform + 0.5*resampling)
    
    #transform time coordinate from [-1,1] to [sd,1]
    samples = samples.at[:,0].add(1.0+2*sd)
    samples = samples.at[:,0].divide(2.0+2*sd)
    return ([s.reshape(-1,1) for s in samples.T], sample_probability_weight) # return samples, colloc_weights)


