# tiny-muon-jax

```python
from muon import muon

optimizer = muon()

opt_state = optimizer.init(params)

updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```