"""Tiny Muon implementation.

Allows for any sharding arrangement (just shard mu and nu same as params) and uses vmap/lax.map 
for scanned layers.
"""

from typing import Callable, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import optax


def ns(x):
    if x.shape[-2] > x.shape[-1]:
        x = x.T
    x /= jnp.linalg.norm(x) + 1e-7
    for _ in range(5):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    if x.shape[-2] < x.shape[-1]:
        x = x.T
    return x


class MuonState(NamedTuple):
    count: jax.Array
    mu: jax.Array
    nu: jax.Array


def muon(
    learning_rate: optax.ScalarOrSchedule = 0.01,
    b1: float = 0.95,
    weight_decay: float = 1e-5,
    adam_learning_rate: optax.ScalarOrSchedule = 3e-4,
    adam_b1: float = 0.9,
    adam_b2: float = 0.99,
    adam_eps: float = 1e-8,
    adam_weight_decay: float = 0.0,
    scan_layer: Callable[[str, jax.Array], bool] = lambda path, x: "scan" in path.lower(),
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    use_adam: Callable[[str, jax.Array], bool] = lambda path, x: False,
    momentum_dtype: jnp.dtype = jnp.bfloat16,
    newton_schulz_dtype: jnp.dtype = jnp.bfloat16,
) -> optax.GradientTransformation:
    """Tiny Muon implementation.

    Allows for any sharding arrangement (just shard mu and nu same as params) and uses vmap/lax.map 
    for scanned layers.

    Args:
        learning_rate: learning rate
        b1: momentum
        weight_decay: weight decay
        adam_learning_rate: learning rate for adam
        adam_b1: momentum for adam
        adam_b2: beta2 for adam
        adam_eps: epsilon for adam
        adam_weight_decay: weight decay for adam
        scan_layer: fn that takes in keys joined with '/' and the param array and returns a bool
        lax_map_scanned_layers: whether to use lax.map for scanned layers instead of vmap
        lax_map_batch_size: batch size for jax.lax.map
        use_adam: fn that takes in keys joined with '/' and the param array and returns a bool (ndim < 2 always True)
        momentum_dtype: dtype for momentum
        newton_schulz_dtype: dtype for newton schulz iterations

    Returns:
        optax.GradientTransformation
    """
    map_fn = lambda fn, *xs: (
        jax.lax.map(lambda xs: fn(*xs), xs, batch_size=lax_map_batch_size if lax_map_batch_size > 1 else None)
        if lax_map_scanned_layers else jax.vmap(fn)(*xs)
    )

    def get_scanned_and_adam_layers(xs):
        scanned_layers = jax.tree.map_with_path(
            lambda path, x: scan_layer(jax.tree_util.keystr(path, simple=True, separator="/"), x), xs
        )
        adam_layers = jax.tree.map_with_path(
            lambda path, x, s: (
                len(x.shape[s:]) < 2 or np.prod(x.shape[s:]) == max(x.shape[s:])
                or use_adam(jax.tree_util.keystr(path, simple=True, separator="/"), x)
            ),
            xs, scanned_layers,
        )
        return scanned_layers, adam_layers

    def init_fn(params):
        _, adam_layers = get_scanned_and_adam_layers(params)
        return MuonState(
            count=jnp.zeros([], dtype=jnp.int32),
            mu=jax.tree.map(lambda x: jnp.zeros_like(x, dtype=momentum_dtype), params),
            nu=jax.tree.map(lambda x, a: jnp.zeros_like(x, dtype=jnp.float32) if a else None, params, adam_layers),
        )

    def muon_update(updates, mu, nu):
        trace_fn = lambda g, t: g + b1 * t
        mu = trace_fn(updates, mu)
        updates = ns(reshape_to_matrix(trace_fn(updates, mu)).astype(newton_schulz_dtype)).astype(mu.dtype).reshape(mu.shape)
        return updates * jnp.sqrt(jnp.maximum(1, updates.shape[-2] / updates.shape[-1])), mu.astype(momentum_dtype), nu

    def adam_update(updates, mu, nu, count):
        mu, nu = adam_b1 * mu + (1 - adam_b1) * updates, adam_b2 * nu + (1 - adam_b2) * updates**2
        mu_hat, nu_hat = mu / (1 - adam_b1**count), nu / (1 - adam_b2**count)
        return mu_hat / (jnp.sqrt(nu_hat) + adam_eps), mu.astype(momentum_dtype), nu

    def update_fn(updates, state, params=None):
        assert params is not None, optax._src.base.NO_PARAMS_MSG
        scanned_layers, adam_layers = get_scanned_and_adam_layers(updates)
        count = optax.safe_int32_increment(state.count)
        outputs = jax.tree.map(
            lambda u, m, n, s, a: (
                adam_update(u, m, n, count) if a else (map_fn(muon_update, u, m, n) if s else muon_update(u, m, n))
            ),
            updates, state.mu, state.nu, scanned_layers, adam_layers,
        )
        updates, mu, nu = [
            jax.tree.unflatten(jax.tree.structure(state.mu), x)
            for x in list(zip(*jax.tree.structure(state.mu).flatten_up_to(outputs)))
        ]
        updates = jax.tree.map(lambda u, p, a: u + (adam_weight_decay if a else weight_decay) * p, updates, params, adam_layers)
        lr = learning_rate(count) if callable(learning_rate) else learning_rate
        adam_lr = adam_learning_rate(count) if callable(adam_learning_rate) else adam_learning_rate
        lr_tree = jax.tree.map(lambda _, a: adam_lr if a else lr, params, adam_layers)
        updates = jax.tree.map(lambda p, u, lr: p - lr * u, params, updates, lr_tree)
        return updates, MuonState(count, mu, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def reshape_to_matrix(x):
    squash_left = [np.prod(x.shape[:-1], dtype=np.int32), x.shape[-1]]
    squash_right = [x.shape[0], np.prod(x.shape[1:], dtype=np.int32)]
    return x.reshape(squash_left if np.abs(np.diff(squash_left)) < np.abs(np.diff(squash_right)) else squash_right)
