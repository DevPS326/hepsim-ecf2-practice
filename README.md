# Standalone jet observables: normalized ECF2

An independent, small Python implementation of the normalized two-point
energy correlation function for **jet physics practice**, motivated by
ML4SCI HEPSIM's jet-observable-library project idea. **Not an official
ML4SCI repository or submission**, and not yet reviewed or accepted by mentors.

## Definition

For one jet with constituent transverse momenta `pT`, rapidities `y`, and
azimuthal angles `phi`, the normalized observable is

$$
e_2^{(\beta)} = \sum_{i<j} z_i z_j (\Delta R_{ij})^\beta,
\qquad z_i=\frac{p_{T,i}}{\sum_k p_{T,k}}, \quad \beta > 0.
$$

`Delta R = sqrt((y_i - y_j)^2 + wrapped(phi_i - phi_j)^2)` with phi
differences wrapped to `[-pi, pi]`. Every **unordered pair appears once**
(`i < j`). No grooming or extra jet-radius normalization is applied.
Different physics libraries sometimes use ordered pairs (factor of two)
or alternate energy/angular conventions: compare like with like.

This observable is dimensionless; PDG ID is ignored. The coordinate is
**rapidity**, not pseudorapidity. The function accepts exactly four columns.

## Installation and example

Python 3.9+ and NumPy are required. From the extracted folder:

```bash
python -m pip install .
python -m examples.demo_ecf2
python -m unittest discover -s tests -v
```

Or use the API directly:

```python
import numpy as np
from jet_observables import ecf2

jet = np.array([
    [1.0, 0.0, 0.0, 211],
    [1.0, 1.0, 0.0, -211],
    [0.0, 0.0, 0.0, 0],  # zero-padded row
])

assert np.isclose(ecf2(jet, beta=1.0), 0.25)
print(ecf2(jet))  # 0.25
```

Input shape: `(N, 4)`, columns `[pT, y, phi, PDG ID]`.
All values must be finite; negative pT is rejected, and zero-pT rows are
ignored. `beta` must be finite and positive. Fewer than two constituents with
positive pT yield `0.0`. The implementation scales as O(N²) in memory/time.

## Verification and limitations

- The dataset-free regression suite covers analytical results, phi wrapping,
  invariances, exact collinear splitting, the soft-particle limit, input
  validation, and extreme pT normalization. A separate slow nested-loop
  calculation checks randomized synthetic jets at multiple `beta` values.
- An earlier separate Colab exercise applied ECF2 to 100 Pythia8 quark/gluon
  dataset jets. That dataset is **not included** and the standalone tests
  **do not** claim to rerun the original data analysis.
- A numerical comparison with an **installed** independent third-party
  implementation such as EnergyFlow or FastJet has not yet been run. This
  module has not undergone detector or cross-generator physics validation.
- This is a single-observable practice package, not a mature multi-observable
  ML4SCI library, and no project acceptance is implied.

## References

- [ML4SCI HEPSIM2 2026 project idea](https://ml4sci.org/gsoc/2026/proposal_HEPSIM2.html)
- [EnergyFlow hadronic measure definitions](https://energyflow.network/docs/measures/)
- [FastJet-related talk showing normalized e2](https://soyez.fastjet.fr/data/talks/2016-04-ULg.pdf)
