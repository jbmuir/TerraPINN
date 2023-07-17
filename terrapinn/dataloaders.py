import jax
import jax.numpy as jnp
from sklearn.neighbors import KernelDensity


def weighted_dataloader(rng_key, *data, weights=None, num_batches=30000, batch_size=10000):
    split_rng_key = jax.random.split(rng_key, num_batches)
    def _dset_generator():
        for i in range(num_batches):
            idx = jax.random.choice(split_rng_key[i], len(data[0]), (batch_size,))
            yield [d[idx] for d in data]
    
    return _dset_generator()

def kde_resampler(rng_key, coords, weights, n_resample, bandwidth=0.2):
    kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
    transformed_coords = jnp.arctanh(coords)
    transform_jacobian = jnp.prod(1/(1-jnp.square(coords)), axis=1)
    kde.fit(transformed_coords, sample_weight=weights/transform_jacobian)
    resampled_coords = kde.sample(n_resample, random_state=int(rng_key[1]))
    untransformed_resampled_coords = jnp.tanh(resampled_coords)
    return untransformed_resampled_coords

def uniform_plus_kde_sampler(rng_key, ndims, weights=None, old_coords=None, hbatch_size=10000, bandwidth=0.2):
    rng_key_1, rng_key_2 = jax.random.split(rng_key)
    sample_half_1 = jax.random.uniform(rng_key_1, (hbatch_size, ndims))
    # turn [0,1] to [-1,1] for all but the first dimension using JAX jit-inbound-guaranteed notation
    sample_half_1 = sample_half_1.at[:,1:].multiply(2)
    sample_half_1 = sample_half_1.at[:,1:].add(-1)
    if weights == None:
        sample_half_2 = jax.random.uniform(rng_key_2, (hbatch_size, ndims))
        sample_half_2 = sample_half_2.at[:,1:].multiply(2)
        sample_half_2 = sample_half_2.at[:,1:].add(-1)
    else:
        sample_half_2 = kde_resampler(rng_key_2, old_coords, weights, hbatch_size, bandwidth=bandwidth)

    return jnp.vstack((sample_half_1, sample_half_2))