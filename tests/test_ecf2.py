"""Dataset-free analytical and regression checks for normalized ECF2."""

import unittest

import numpy as np

from jet_observables import ecf2


def slow_reference_ecf2(particles, beta=1.0):
    """Independent nested-loop definition for ordinary, well-scaled inputs."""
    active = np.asarray(particles, dtype=float)
    active = active[active[:, 0] > 0]
    if len(active) < 2:
        return 0.0
    total_pt = sum(float(row[0]) for row in active)
    result = 0.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            dy = float(active[i, 1] - active[j, 1])
            dphi = float(active[i, 2] - active[j, 2])
            dphi = ((dphi + np.pi) % (2 * np.pi)) - np.pi
            dr = np.hypot(dy, dphi)
            result += (active[i, 0] / total_pt) * (active[j, 0] / total_pt) * dr**beta
    return float(result)


class TestECF2(unittest.TestCase):
    def setUp(self):
        self.jet = np.array([
            [10.0, 0.2, 1.0, 211],
            [5.0, -0.1, -0.5, -211],
            [2.0, 0.8, 2.5, 22],
            [0.0, 0.0, 0.0, 0],
        ])

    def test_analytic_two_particle_beta_1(self):
        self.assertAlmostEqual(ecf2([[1, 0, 0, 211], [1, 1, 0, -211]]), 0.25, places=14)

    def test_analytic_two_particle_beta_2(self):
        self.assertAlmostEqual(ecf2([[1, 0, 0, 211], [1, 2, 0, -211]], 2), 1.0, places=14)

    def test_empty_and_single(self):
        self.assertEqual(ecf2(np.empty((0, 4))), 0.0)
        self.assertEqual(ecf2([[5, 0, 0, 211]]), 0.0)
        self.assertEqual(ecf2([[0, 0, 0, 0], [0, 1, 2, 0]]), 0.0)

    def test_phi_periodicity_at_boundary(self):
        jet = [[1, 0, np.pi - 0.01, 211], [1, 0, -np.pi + 0.01, -211]]
        self.assertAlmostEqual(ecf2(jet), 0.005, places=12)

    def test_phi_rotation_invariance(self):
        rotated = self.jet.copy()
        rotated[:, 2] = (rotated[:, 2] + 1.5) % (2 * np.pi)
        self.assertAlmostEqual(ecf2(self.jet), ecf2(rotated), places=12)

    def test_rapidity_shift_invariance(self):
        shifted = self.jet.copy()
        shifted[:, 1] += 3.5
        self.assertAlmostEqual(ecf2(self.jet), ecf2(shifted), places=12)

    def test_permutation_invariance(self):
        self.assertAlmostEqual(ecf2(self.jet), ecf2(self.jet[[2, 0, 3, 1]]), places=12)

    def test_pt_scale_invariance(self):
        scaled = self.jet.copy()
        scaled[:, 0] *= 12.5
        self.assertAlmostEqual(ecf2(self.jet), ecf2(scaled), places=12)

    def test_padding_invariance(self):
        extra = np.array([[0, 9, 8, 0], [0, -4, -2, 0]])
        self.assertAlmostEqual(ecf2(self.jet), ecf2(np.vstack([self.jet, extra])), places=12)

    def test_pid_invariance(self):
        changed = self.jet.copy()
        changed[:, 3] = 13
        self.assertAlmostEqual(ecf2(self.jet), ecf2(changed), places=12)

    def test_collinear_splitting(self):
        split = np.array([
            [4, 0.2, 1.0, 211], [6, 0.2, 1.0, 211],
            [5, -0.1, -0.5, -211], [2, 0.8, 2.5, 22],
        ])
        self.assertAlmostEqual(ecf2(self.jet), ecf2(split), places=12)

    def test_soft_particle_limit(self):
        softened = np.vstack([self.jet, [1e-10, 0.6, -2, 11]])
        self.assertAlmostEqual(ecf2(self.jet), ecf2(softened), delta=1e-9)

    def test_nonfinite_beta_rejected(self):
        for beta in (np.nan, np.inf, -np.inf, 0.0, -1.0, "1", True, 1j, np.array([1.])):
            with self.subTest(beta=str(beta)), self.assertRaises(ValueError):
                ecf2(self.jet, beta=beta)

    def test_invalid_particle_values_and_shape(self):
        invalid_inputs = (
            [[1, 2, 3]], [[1, 2, 3, 4, 5]], [1, 2, 3, 4],
            [[-1, 0, 0, 211]], [[np.nan, 0, 0, 211]],
            [[1, np.inf, 0, 211]], [[1, 0, np.nan, 211]],
            [[1, 0, 0, np.nan]], [["bad", 0, 0, 211]],
        )
        for invalid in invalid_inputs:
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                ecf2(invalid)

    def test_large_float_pt_no_overflow(self):
        jet = [[1e308, 0, 0, 211], [1e308, 1, 0, -211]]
        self.assertAlmostEqual(ecf2(jet), 0.25, places=14)

    def test_large_integer_pt_no_overflow(self):
        jet = np.array([
            [2**62, 0, 0, 211], [2**62, 1, 0, -211], [2**62, 2, 0, 22],
        ], dtype=np.int64)
        self.assertAlmostEqual(ecf2(jet), 4 / 9, places=14)

    def test_tiny_pt_no_underflow(self):
        jet = [[1e-300, 0, 0, 211], [1e-300, 1, 0, -211]]
        self.assertAlmostEqual(ecf2(jet), 0.25, places=14)

    def test_reference_random_jets(self):
        rng = np.random.default_rng(2027)
        for beta in (0.5, 1.0, 2.0):
            for count in (2, 3, 10, 25):
                jet = np.column_stack([
                    rng.uniform(0.1, 50, count), rng.normal(0, 1, count),
                    rng.uniform(-4, 4, count), np.full(count, 211),
                ])
                with self.subTest(beta=beta, n=count):
                    self.assertAlmostEqual(ecf2(jet, beta), slow_reference_ecf2(jet, beta), places=12)

    def test_nonfinite_geometry_rejected(self):
        jet = [[1, 1e308, 0, 211], [1, -1e308, 0, -211]]
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaises(ValueError):
                ecf2(jet)


if __name__ == "__main__":
    unittest.main()
