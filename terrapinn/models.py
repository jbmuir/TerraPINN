import haiku as hk
import jax

def default_radial_model(layers=(32,32), output=1):
    def radial_model_fn(tr):
        model_list = []
        for layer in layers:
            model_list += [hk.Linear(layer), jax.nn.swish]
        
        model_list += [hk.Linear(output)]
        mlp = hk.Sequential(model_list)
        return mlp(tr)

    return hk.without_apply_rng(hk.transform(radial_model_fn))

def default_2d_model():
    def amp_tau_model_fn(txy):
        mlp = hk.Sequential([
        hk.Linear(64), jax.nn.swish,
        hk.Linear(32), jax.nn.swish,
        hk.Linear(32), jax.nn.swish,
        hk.Linear(16), jax.nn.swish,
        hk.Linear(4, with_bias=False), # output all a, t, r, dk components 
        ])
        return mlp(txy)

    return hk.without_apply_rng(hk.transform(amp_tau_model_fn))