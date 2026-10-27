"""Numerical benchmark of normalized ECF2 against EnergyFlow's EFP reference.

EnergyFlow is an optional, test-only dependency: the benchmark is skipped when
it is not installed (see the README; EnergyFlow 1.4.0 needs NumPy < 2).

Convention. ``energyflow.EFP([(0, 1)], measure="hadr", normed=True,
coords="ptyphim")`` evaluates the *ordered* double sum

    sum_i sum_j z_i z_j (dy_ij**2 + dphi_ij**2)**(beta / 2),
    z_i = pT_i / sum_k pT_k,

so every unordered pair is counted twice, whereas ``ecf2`` counts each
unordered pair once: ``ecf2(jet, beta) == 0.5 * EFP``.
"""

import unittest

import numpy as np

from jet_observables import ecf2

try:
    import energyflow
except ModuleNotFoundError as exc:  # only absence skips; a broken install should fail loudly
    energyflow = None
    MISSING = str(exc)
else:
    MISSING = ""

BETAS = (0.5, 1.0, 2.0)
RTOL = 1e-12  # float64 round-off on these sums is ~1e-15; leaves ~3 orders of margin
JET_SEED = 20261027
N_CONSTITUENTS = (2, 3, 5, 8, 13, 21, 34, 55)
PDG_IDS = (211, -211, 22, 130, 2212, -2212, 11, -13)


def wrap_signed(phi):
    """Map azimuth into (-pi, pi]."""
    return np.pi - np.mod(np.pi - phi, 2 * np.pi)


def wrap_positive(phi):
    """Map azimuth into [0, 2pi)."""
    return np.mod(phi, 2 * np.pi)


def make_synthetic_jets(seed=JET_SEED):
    """Deterministic jets ``[pT, y, phi, PDG ID]``: 8 multiplicities x 3 azimuth set-ups.

    Set-ups: a jet in the interior of (-pi, pi]; a jet on the +-pi cut in
    (-pi, pi]; a jet on the 0/2pi cut in [0, 2pi). In the two cut set-ups the
    constituents alternate sides of the cut, so every such jet has pairs whose
    raw phi difference is close to 2pi and only the periodic wrap gives the
    correct angular distance.
    """
    rng = np.random.default_rng(seed)
    setups = ((0.4, wrap_signed, False), (np.pi, wrap_signed, True), (0.0, wrap_positive, True))
    jets = []
    for axis_phi, wrap, straddle in setups:
        for n in N_CONSTITUENTS:
            pt = np.exp(rng.uniform(np.log(0.2), np.log(200.0), n))
            y = rng.uniform(-2.0, 2.0) + 0.3 * rng.standard_normal(n)
            dphi = 0.3 * rng.standard_normal(n)
            if straddle:
                dphi = np.abs(dphi) * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
            pdg = np.resize(PDG_IDS, n)
            jets.append(np.column_stack([pt, y, wrap(axis_phi + dphi), pdg]))
    return jets


def energyflow_event(jet):
    """EnergyFlow ``ptyphim`` event ``[pT, y, phi, mass]``: the PDG ID column becomes mass 0."""
    event = np.array(jet, dtype=np.float64)  # copy, so the caller's jet is untouched
    event[:, 3] = 0.0
    return event


def unordered_from_ordered(ordered_sum):
    """EFP sums (i, j) and (j, i); ecf2 sums each unordered pair once."""
    return 0.5 * ordered_sum


class TestSyntheticJetSet(unittest.TestCase):
    """Properties of the benchmark input itself; needs no EnergyFlow."""

    def test_at_least_twenty_deterministic_jets(self):
        jets = make_synthetic_jets()
        self.assertGreaterEqual(len(jets), 20)
        for first, second in zip(jets, make_synthetic_jets()):
            np.testing.assert_array_equal(first, second)

    def test_jets_are_valid_input_with_nonzero_pdg_ids(self):
        for index, jet in enumerate(make_synthetic_jets()):
            with self.subTest(jet=index):
                self.assertEqual(jet.ndim, 2)
                self.assertEqual(jet.shape[1], 4)
                self.assertGreaterEqual(len(jet), 2)
                self.assertTrue(np.isfinite(jet).all())
                self.assertTrue((jet[:, 0] > 0).all())
                self.assertTrue((jet[:, 3] != 0).all())  # so replacing PDG IDs by mass 0 matters

    def test_cut_setups_exercise_periodic_phi_wrap(self):
        straddling = [jet for jet in make_synthetic_jets() if np.ptp(jet[:, 2]) > np.pi]
        self.assertGreaterEqual(len(straddling), 16)


@unittest.skipIf(
    energyflow is None,
    f"EnergyFlow is not importable ({MISSING}); see README, 'Benchmark against EnergyFlow'",
)
class TestEnergyFlowReference(unittest.TestCase):
    report = []

    @classmethod
    def setUpClass(cls):
        cls.efps = {
            beta: energyflow.EFP([(0, 1)], measure="hadr", beta=beta, normed=True, coords="ptyphim")
            for beta in BETAS
        }

    @classmethod
    def tearDownClass(cls):
        if cls.report:
            print("\n" + "\n".join(cls.report), flush=True)

    def test_efp_configuration_matches_documented_convention(self):
        for beta, efp in self.efps.items():
            with self.subTest(beta=beta):
                self.assertEqual(efp.measure, "hadr")
                self.assertEqual(efp.coords, "ptyphim")
                self.assertTrue(efp.normed)
                self.assertEqual(efp.kappa, 1)
                self.assertEqual(efp.beta, beta)
                self.assertEqual(efp.graph, [(0, 1)])

    def test_energyflow_event_replaces_pdg_ids_with_zero_mass(self):
        jet = np.array([[3.0, 0.1, 0.2, 211], [1.0, -0.3, 0.4, -13]])
        original = jet.copy()
        event = energyflow_event(jet)
        np.testing.assert_array_equal(event[:, :3], original[:, :3])
        np.testing.assert_array_equal(event[:, 3], [0.0, 0.0])
        np.testing.assert_array_equal(jet, original)

    def test_ordered_pair_factor_of_two_on_analytic_jets(self):
        # Two equal-pT constituents at separation d have z = 1/2 each, so the
        # unordered e2 is (1/2)(1/2) d**beta; EFP also counts (j, i): twice that.
        for beta, separation in ((1.0, 1.0), (2.0, 2.0), (0.5, 4.0)):
            jet = [[1.0, 0.0, 0.0, 211], [1.0, separation, 0.0, -211]]
            unordered = 0.25 * separation**beta
            with self.subTest(beta=beta):
                ordered = self.efps[beta].compute(energyflow_event(jet))
                self.assertAlmostEqual(ecf2(jet, beta), unordered, places=14)
                self.assertAlmostEqual(ordered, 2 * unordered, places=14)
                self.assertAlmostEqual(unordered_from_ordered(ordered), unordered, places=14)

    def test_matches_energyflow_on_synthetic_jets(self):
        jets = make_synthetic_jets()
        self.assertGreaterEqual(len(jets), 20)
        worst = {}
        for beta in BETAS:
            max_abs = max_rel = 0.0
            for index, jet in enumerate(jets):
                value = ecf2(jet, beta)
                reference = unordered_from_ordered(self.efps[beta].compute(energyflow_event(jet)))
                self.assertGreater(reference, 0.0)
                abs_err = abs(value - reference)
                rel_err = abs_err / reference
                max_abs, max_rel = max(max_abs, abs_err), max(max_rel, rel_err)
                with self.subTest(beta=beta, jet=index, n=len(jet)):
                    self.assertLessEqual(
                        rel_err, RTOL,
                        f"ecf2={value!r} vs 0.5*EFP={reference!r} (abs err {abs_err:.3e})",
                    )
            worst[beta] = (max_abs, max_rel)

        type(self).report = [
            f"EnergyFlow {energyflow.__version__} reference benchmark (NumPy {np.__version__}): "
            f"{len(jets)} jets x {len(BETAS)} betas = {len(jets) * len(BETAS)} comparisons, "
            f"ecf2 vs 0.5 * EFP, rtol {RTOL:g}",
            "  beta   max abs error   max rel error",
            *(f"  {beta:<5.1f} {worst[beta][0]:>13.3e} {worst[beta][1]:>15.3e}" for beta in BETAS),
            f"  all   {max(w[0] for w in worst.values()):>13.3e} "
            f"{max(w[1] for w in worst.values()):>15.3e}",
        ]

    def test_zero_pt_padding_and_single_particle_agree(self):
        padded = [
            [10.0, 0.2, 1.0, 211], [5.0, -0.1, -0.5, -211], [2.0, 0.8, 2.5, 22],
            [0.0, 0.0, 0.0, 0], [0.0, 3.0, -2.0, 0],
        ]
        single = [[5.0, 0.1, 0.2, 211]]
        for beta in BETAS:
            for name, jet in (("padded", padded), ("single", single)):
                with self.subTest(beta=beta, jet=name):
                    reference = unordered_from_ordered(self.efps[beta].compute(energyflow_event(jet)))
                    self.assertAlmostEqual(ecf2(jet, beta), reference, places=14)


if __name__ == "__main__":
    unittest.main()
