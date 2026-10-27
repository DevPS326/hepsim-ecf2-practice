"""Reproducible ECF2 example without dataset downloads."""

import numpy as np

from jet_observables import ecf2


jet = np.array([
    [1.0, 0.0, 0.0, 211],
    [1.0, 1.0, 0.0, -211],
    [0.0, 0.0, 0.0, 0],
])
value = ecf2(jet, beta=1.0)
assert np.isclose(value, 0.25)
print(f"Normalized e2(beta=1): {value:.6f}")
