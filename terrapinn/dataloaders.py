import jax
import jax.numpy as jnp
from sklearn.neighbors import KernelDensity


def weighted_dataloader(dset_rng_key, *data, weights=None, num_batches=30000, batch_size=10000):
    split_rng_key = jax.random.split(dset_rng_key, num_batches)
    def _dset_generator():
        for i in range(num_batches):
            idx = jax.random.choice(split_rng_key[i], len(data[0]), (batch_size,))
            yield [d[idx] for d in data]
    
    return _dset_generator()

def kde_resampler(coords, weights, n_resample, bandwidth=0.2):
    kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
    transformed_coords = jnp.arctanh(coords)
    transform_jacobian = jnp.prod(1/(1-jnp.square(coords)), axis=1)
    kde.fit(transformed_coords, sample_weight=weights/transform_jacobian)
    resampled_coords = kde.sample(n_resample)
    untransformed_resampled_coords = jnp.tanh(resampled_coords)
    return untransformed_resampled_coords