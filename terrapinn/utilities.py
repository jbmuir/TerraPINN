import jax

def pytree_size(pytree):
    """Count the total number of parameters in a JAX PyTree"""
    return jax.tree_util.tree_reduce(lambda x, y: x+y, jax.tree_map(lambda x:x.size, pytree), 0)
