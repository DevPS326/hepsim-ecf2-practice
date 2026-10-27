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
EnergyFlow's single-edge `EFP` is such an ordered-pair case; see
[Benchmark against EnergyFlow](#benchmark-against-energyflow).

This observable is dimensionless; PDG ID is ignored. The coordinate is
**rapidity**, not pseudorapidity. The function accepts exactly four columns.

## Installation and example

Python 3.9+ and NumPy are required. From the extracted folder:

```bash
python -m pip install .
python -m examples.demo_ecf2
python -m unittest discover -s tests -v
```

The EnergyFlow reference tests in `tests/test_energyflow_reference.py` are
optional: they are skipped, with an explicit message, unless EnergyFlow is
importable. See [Benchmark against EnergyFlow](#benchmark-against-energyflow)
for the separate (NumPy < 2) environment they need.

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
- An independent numerical comparison with **installed** EnergyFlow 1.4.0 on
  synthetic jets is documented in
  [Benchmark against EnergyFlow](#benchmark-against-energyflow). A comparison
  with FastJet has **not** been run, and this module has not undergone
  detector or cross-generator physics validation.
- This is a single-observable practice package, not a mature multi-observable
  ML4SCI library, and no project acceptance is implied.

## Benchmark against EnergyFlow

`tests/test_energyflow_reference.py` compares `ecf2()` with
[EnergyFlow](https://energyflow.network/), an independent third-party
implementation, on 24 deterministic synthetic jets (2 to 55 constituents,
fixed seed `20261027`; 16 of them straddle an azimuthal periodic cut) at
`beta = 0.5, 1.0, 2.0`, i.e. 72 comparisons.

**Reference version (exact).** EnergyFlow **1.4.0** (`energyflow.__version__`),
installed from PyPI with its declared dependencies: h5py 3.16.0,
wasserstein 1.1.0, wurlitzer 3.1.1 and NumPy 1.26.4, on Python 3.11.9
(Windows 11, 64-bit).

**EnergyFlow call.** The default `kappa=1` is used (the tests assert it):

```python
energyflow.EFP([(0, 1)], measure="hadr", beta=beta, normed=True, coords="ptyphim")
```

**Normalization convention.** In EnergyFlow 1.4.0 this single-edge graph is
evaluated as an *ordered* double sum over all index pairs (an `einsum` of
`'ab,a,b'` with no symmetry factor):

$$
\mathrm{EFP}_{[(0,1)]} = \sum_{i=1}^{M}\sum_{j=1}^{M} z_i z_j\,\theta_{ij}^{\beta}
= 2\,e_2^{(\beta)},
\qquad z_i=\frac{p_{T,i}}{\sum_k p_{T,k}},
\quad \theta_{ij}=\sqrt{\Delta y_{ij}^2+\Delta\phi_{ij}^2},
$$

with `Delta phi` the periodic azimuthal difference. `EFP` therefore counts
every unordered pair twice (as `(i, j)` and `(j, i)`) while `ecf2()` counts it
once, so the tests compare `ecf2(jet, beta)` with **`0.5 * EFP`**. The factor of
two is also pinned to a hand-calculable case: two equal-`pT` constituents one
rapidity unit apart give `ecf2 = 0.25` and `EFP = 0.5` at `beta = 1`.

**Input mapping.** `ecf2()` rows `[pT, y, phi, PDG ID]` become EnergyFlow
`ptyphim` rows `[pT, y, phi, mass]` with the PDG ID column **replaced by mass
0**. (For `measure="hadr"`, EnergyFlow 1.4.0 reads only the first three
columns, so mass does not enter the value; it is still set to 0 instead of
passing a PDG ID as a "mass".)

**Results.** Absolute error is `|ecf2 - 0.5*EFP|` and relative error is the
absolute error divided by `0.5*EFP`; each entry is the maximum over the 24
jets.

| beta | max abs error | max rel error |
|-----:|--------------:|--------------:|
| 0.5  | 2.220e-16     | 9.651e-16     |
| 1.0  | 1.943e-16     | 8.979e-16     |
| 2.0  | 2.220e-16     | 1.294e-15     |
| all  | 2.220e-16     | 1.294e-15     |

This is float64 round-off level (`2.220e-16` is one unit in the last place for
values near 1). The test tolerance is a relative error of `1e-12`, and the test
prints this table when it runs.

**Environment.** EnergyFlow 1.4.0 requires `wasserstein>=1.0.1`; the
`wasserstein` 1.1.0 wheel (a compiled extension) declares `numpy<2.0.0`, so
`pip install energyflow` resolves to NumPy 1.26.4. That would downgrade an
existing NumPy 2.x installation (NumPy 2.4.4 on the machine used here), so
EnergyFlow was installed into a separate virtual environment instead:

```bash
python -m venv ../energyflow-venv
../energyflow-venv/Scripts/python -m pip install energyflow==1.4.0   # POSIX: bin/python
../energyflow-venv/Scripts/python -m unittest discover -s tests -v   # from this folder
```

Without EnergyFlow, the five EnergyFlow-dependent tests are skipped (the skip
message names the missing module); all other tests are unaffected.

**Scope.** This checks numerical agreement of one observable with one
independent implementation on synthetic jets. It is not a comparison with
FastJet, real events, or detector effects.

## References

- [ML4SCI HEPSIM2 2026 project idea](https://ml4sci.org/gsoc/2026/proposal_HEPSIM2.html)
- [EnergyFlow hadronic measure definitions](https://energyflow.network/docs/measures/)
- [FastJet-related talk showing normalized e2](https://soyez.fastjet.fr/data/talks/2016-04-ULg.pdf)
