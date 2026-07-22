import unittest

import numpy as np

from utils.uncertainty_quantification import (
    DPkP,
    E_M,
    EPR,
    EPRD,
    EPRDS,
    M_E,
    Max_E,
    entropy,
    misclassification_probability,
)


class UncertaintyQuantificationTests(unittest.TestCase):
    def setUp(self):
        # Four stochastic observations, three samples, three classes.
        self.obs = np.array(
            [
                [[0.80, 0.10, 0.10], [0.40, 0.35, 0.25], [0.10, 0.20, 0.70]],
                [[0.75, 0.15, 0.10], [0.36, 0.39, 0.25], [0.15, 0.15, 0.70]],
                [[0.85, 0.10, 0.05], [0.45, 0.30, 0.25], [0.10, 0.25, 0.65]],
                [[0.78, 0.12, 0.10], [0.34, 0.41, 0.25], [0.12, 0.18, 0.70]],
            ],
            dtype=float,
        )

    def test_every_method_returns_one_finite_score_per_sample(self):
        mp, mp_mean = misclassification_probability(self.obs)
        methods = [
            mp,
            mp_mean,
            entropy(self.obs),
            M_E(self.obs),
            E_M(self.obs),
            Max_E(self.obs),
            DPkP(self.obs),
            EPR(self.obs),
            EPRD(self.obs),
            EPRDS(self.obs),
        ]
        for values in methods:
            self.assertEqual(values.shape, (3,))
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all(values >= 0.0))
            self.assertTrue(np.all(values <= 1.0 + 1e-12))

    def test_eprds_is_mean_epr_scaled_by_normalized_deviation(self):
        epr = EPR(self.obs)
        deviation = EPRD(self.obs)
        np.testing.assert_allclose(EPRDS(self.obs), epr * deviation)

    def test_no_stochastic_variation_gives_zero_deviation_methods(self):
        deterministic = np.repeat(self.obs[:1], repeats=5, axis=0)
        np.testing.assert_allclose(DPkP(deterministic), 0.0, atol=1e-12)
        np.testing.assert_allclose(EPRD(deterministic), 0.0, atol=1e-12)
        np.testing.assert_allclose(EPRDS(deterministic), 0.0, atol=1e-12)

    def test_binary_class_epr_matches_categorical_entropy_per_observation(self):
        binary = np.array(
            [
                [[0.8, 0.2], [0.55, 0.45]],
                [[0.7, 0.3], [0.60, 0.40]],
            ]
        )
        expected = np.mean(
            [entropy(binary[index : index + 1]) for index in range(binary.shape[0])],
            axis=0,
        )
        np.testing.assert_allclose(EPR(binary), expected)

    def test_invalid_shape_raises_clear_error(self):
        with self.assertRaises(ValueError):
            EPRDS(np.ones((3, 4)))


if __name__ == "__main__":
    unittest.main()
