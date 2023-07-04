import haiku as hk
import jax

def default_mlp_model(layers, activation_fn = jax.nn.swish, output = 1, output_bias = True):
    def model_fn(x):
        model_list = []
        for layer in layers:
            model_list += [hk.Linear(layer), activation_fn]
        
        model_list += [hk.Linear(output, with_bias=output_bias)]
        mlp = hk.Sequential(model_list)
        return mlp(x)

    return hk.without_apply_rng(hk.transform(model_fn))
