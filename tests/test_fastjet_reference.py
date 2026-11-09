"""Numerical cross-check of normalized ECF2 against FastJet contrib EnergyCorrelator.

FastJet (the scikit-hep ``fastjet`` package, which bundles fjcontrib) is an
optional, test-only dependency: these tests are skipped when it is not
installed (see the README, "Benchmark against FastJet").

Reference quantity. With its default ``pt_R`` measure, EnergyCorrelator uses
``E_i = pT_i`` and ``theta_ij**2 = (y_i - y_j)**2 + dphi_ij**2`` (rapidity and
periodic azimuth), so

    C1^(beta) = ECF(2, beta) / ECF(1, beta)**2
              = sum_(i<j) pT_i pT_j theta_ij**beta / (sum_k pT_k)**2
              = ecf2(jet, beta).

ECF(1) is the *scalar* sum of constituent pT, not the jet's vector pT.

Conventions the comparison depends on:

* Four-vectors come from ``[pT, y, phi]`` with ``mT = sqrt(pT**2 + m**2)``,
  ``pz = mT sinh(y)`` and ``E = mT cosh(y)``. FastJet then recovers the
  rapidity ``ecf2`` was given, so the result does not depend on the mass
  hypothesis (checked for m = 0 and the charged-pion mass). Building
  ``pz = pT sinh(column 1)`` instead treats rapidity as pseudorapidity, which
  for massive constituents changes FastJet's rapidities (negative control).
* The wrapper evaluates correlators on exclusive jets. Cambridge/Aachen with
  R = 1000 and ``exclusive_jets(1)`` gives one jet holding every constituent:
  ``d_ij = dR_ij**2 / R**2 < 1 = d_iB`` for every pair, so no constituent is
  merged with the beam. The tests assert the constituent count.
* Small-R anti-kT exclusive jets are not a valid reference: constituents are
  merged with the beam and dropped from ``exclusive_jets(1)``, and FastJet
  defines exclusive jets only for kt-like algorithms (negative control).
* Zero-pT padding rows are removed before clustering; ``ecf2`` ignores them.
"""

import unittest

import numpy as np

from jet_observables import ecf2

try:
    import awkward as ak
    import fastjet
except ModuleNotFoundError as exc:  # only absence skips; a broken install should fail loudly
    fastjet = None
    MISSING = str(exc)
else:
    MISSING = ""

BETAS = (0.5, 1.0, 2.0)
PION_MASS = 0.13957039  # GeV, charged pion (PDG)
MASSES = (0.0, PION_MASS)
RTOL = 1e-12  # observed agreement is ~1e-15
JET_SEED = 20261108
N_JETS = 24
N_PADDING = 5
CA_RADIUS = 1000.0


def make_synthetic_jets(seed=JET_SEED, n_jets=N_JETS):
    """Deterministic jets ``[pT, y, phi, PDG ID]`` with 2 to 55 constituents.

    Every jet ends with zero-pT padding rows, and every odd-indexed jet
    straddles phi = +-pi with phi stored in (-pi, pi].
    """
    rng = np.random.default_rng(seed)
    jets = []
    for index in range(n_jets):
        n = int(rng.integers(2, 56))
        axis_phi = np.pi - 0.05 if index % 2 else rng.uniform(-np.pi, np.pi)
        pt = rng.exponential(10.0, n) + 0.3
        y = rng.normal(rng.uniform(-2.0, 2.0), 0.25, n)
        phi = np.angle(np.exp(1j * (axis_phi + rng.normal(0.0, 0.25, n))))
        rows = np.column_stack([pt, y, phi, np.full(n, 211.0)])
        jets.append(np.vstack([rows, np.zeros((N_PADDING, 4))]))
    return jets


def fastjet_particles(jet, mass=0.0, column1="rapidity"):
    """FastJet ``(px, py, pz, E)`` records for the positive-pT rows of ``jet``."""
    rows = np.asarray(jet, dtype=np.float64)
    rows = rows[rows[:, 0] > 0.0]
    pt, coordinate, phi = rows[:, 0], rows[:, 1], rows[:, 2]
    if column1 == "rapidity":
        mt = np.hypot(pt, mass)
        pz, energy = mt * np.sinh(coordinate), mt * np.cosh(coordinate)
    elif column1 == "pseudorapidity":
        pz = pt * np.sinh(coordinate)
        energy = np.sqrt(pt**2 + pz**2 + mass**2)
    else:
        raise ValueError(column1)
    return [
        {"px": float(px), "py": float(py), "pz": float(z), "E": float(e)}
        for px, py, z, e in zip(pt * np.cos(phi), pt * np.sin(phi), pz, energy)
    ]


def one_exclusive_jet(jets, jet_def, mass=0.0, column1="rapidity"):
    events = ak.Array([fastjet_particles(jet, mass, column1) for jet in jets],
                      with_name="Momentum4D")
    return fastjet.ClusterSequence(events, jet_def)


def flat(array):
    return ak.to_numpy(ak.flatten(array, axis=None))


def correlators(sequence, beta):
    """C1, ECF(2)/ECF(1)**2, ECF(1) and constituent count of each exclusive jet."""
    c1 = flat(sequence.exclusive_jets_energy_correlator(njets=1, func="c1", beta=beta))
    ecf_2, ecf_1 = (
        flat(sequence.exclusive_jets_energy_correlator(
            njets=1, func="generic", npoint=npoint, beta=beta, normalized=False))
        for npoint in (2, 1)
    )
    n_constituents = flat(ak.num(sequence.exclusive_jets_constituents(1), axis=-1))
    return c1, ecf_2 / ecf_1**2, ecf_1, n_constituents


def relative_error(value, reference):
    return np.abs(value - reference) / np.abs(reference)


class TestSyntheticJetSet(unittest.TestCase):
    """Properties of the benchmark input itself; needs no FastJet."""

    def test_jets_are_deterministic_and_valid(self):
        jets = make_synthetic_jets()
        self.assertEqual(len(jets), N_JETS)
        for index, (jet, again) in enumerate(zip(jets, make_synthetic_jets())):
            with self.subTest(jet=index):
                np.testing.assert_array_equal(jet, again)
                self.assertTrue(np.isfinite(jet).all())
                self.assertGreaterEqual(int((jet[:, 0] > 0).sum()), 2)
                np.testing.assert_array_equal(jet[-N_PADDING:], 0.0)
                self.assertTrue(np.all(np.abs(jet[:, 2]) <= np.pi))

    def test_half_of_the_jets_straddle_the_phi_cut(self):
        straddling = [jet for jet in make_synthetic_jets() if np.ptp(jet[jet[:, 0] > 0, 2]) > np.pi]
        self.assertGreaterEqual(len(straddling), N_JETS // 2)


@unittest.skipIf(
    fastjet is None,
    f"FastJet is not importable ({MISSING}); see README, 'Benchmark against FastJet'",
)
class TestFastJetReference(unittest.TestCase):
    report = []

    @classmethod
    def setUpClass(cls):
        cls.jets = make_synthetic_jets()
        cls.n_true = np.array([int((jet[:, 0] > 0).sum()) for jet in cls.jets])
        cls.ca = fastjet.JetDefinition(fastjet.cambridge_algorithm, CA_RADIUS)

    @classmethod
    def tearDownClass(cls):
        if cls.report:
            print("\n" + "\n".join(cls.report), flush=True)

    def test_one_exclusive_jet_holds_every_constituent(self):
        for mass in MASSES:
            with self.subTest(mass=mass):
                *_, n_constituents = correlators(one_exclusive_jet(self.jets, self.ca, mass), 1.0)
                np.testing.assert_array_equal(n_constituents, self.n_true)

    def test_analytic_two_particle_jet(self):
        # Equal pT, one rapidity unit apart: z = 1/2 each, so C1 = d**beta / 4.
        jet = [[1.0, 0.0, 0.0, 211], [1.0, 1.0, 0.0, -211]]
        for beta in BETAS:
            with self.subTest(beta=beta):
                c1, *_ = correlators(one_exclusive_jet([jet], self.ca), beta)
                self.assertAlmostEqual(c1[0], 0.25, places=14)
                self.assertAlmostEqual(ecf2(jet, beta), 0.25, places=14)

    def test_ecf1_is_the_scalar_pt_sum(self):
        sequence = one_exclusive_jet(self.jets, self.ca)
        _, _, ecf_1, _ = correlators(sequence, 1.0)
        scalar_sum = np.array([jet[:, 0].sum() for jet in self.jets])
        jets = sequence.exclusive_jets(1)
        vector_pt = np.hypot(flat(jets.px), flat(jets.py))
        np.testing.assert_allclose(ecf_1, scalar_sum, rtol=RTOL)
        self.assertTrue(np.all(vector_pt < scalar_sum * (1 - 1e-6)))

    def test_matches_fastjet_c1_and_ecf_ratio(self):
        lines = []
        for mass in MASSES:
            sequence = one_exclusive_jet(self.jets, self.ca, mass)
            for beta in BETAS:
                ours = np.array([ecf2(jet, beta) for jet in self.jets])
                c1, ratio, _, _ = correlators(sequence, beta)
                err_c1, err_ratio = relative_error(ours, c1), relative_error(ours, ratio)
                with self.subTest(mass=mass, beta=beta):
                    self.assertLessEqual(err_c1.max(), RTOL)
                    self.assertLessEqual(err_ratio.max(), RTOL)
                lines.append(f"  {mass:<9.6f} {beta:<5.1f} {err_c1.max():>14.3e} "
                             f"{err_ratio.max():>16.3e}")
        type(self).report = [
            f"FastJet {fastjet.__version__} (awkward {ak.__version__}, NumPy {np.__version__}) "
            f"reference: {len(self.jets)} jets x {len(BETAS)} betas x {len(MASSES)} masses, "
            f"ecf2 vs EnergyCorrelator C1 and ECF(2)/ECF(1)^2, rtol {RTOL:g}",
            "  mass/GeV  beta  max rel err C1  max rel err ratio",
            *lines,
        ]

    def test_negative_control_small_radius_antikt_drops_constituents(self):
        antikt = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
        c1, _, _, n_constituents = correlators(one_exclusive_jet(self.jets, antikt), 1.0)
        ours = np.array([ecf2(jet, 1.0) for jet in self.jets])
        self.assertTrue(np.any(n_constituents < self.n_true))
        self.assertGreater(relative_error(c1, ours).max(), 1e-3)

    def test_negative_control_pseudorapidity_for_rapidity(self):
        # Soft massive constituents: y and eta differ, so FastJet sees other rapidities.
        soft = [np.column_stack([jet[:, 0] * 0.05, jet[:, 1:]]) for jet in self.jets]
        ours = np.array([ecf2(jet, 1.0) for jet in soft])
        c1, *_ = correlators(
            one_exclusive_jet(soft, self.ca, PION_MASS, column1="pseudorapidity"), 1.0)
        self.assertGreater(relative_error(c1, ours).max(), 1e-6)
        massless, *_ = correlators(one_exclusive_jet(soft, self.ca, 0.0, "pseudorapidity"), 1.0)
        np.testing.assert_allclose(massless, ours, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
