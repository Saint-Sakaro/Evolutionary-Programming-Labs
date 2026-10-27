"""Contract checks: mathematical minimum, constraints, budget and reproducibility."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer import blx_alpha, genetic_algorithm, qing, random_search, reflect


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.problem = {'dimension': 5, 'lower': -500., 'upper': 500.}
        self.parameters = {'population_size': 40, 'budget': 837,
                           'tournament_size': 3, 'crossover_probability': .9,
                           'blx_alpha': .5, 'mutation_probability': .2,
                           'mutation_sigma': .1}

    def test_known_function_values_and_all_minima(self):
        roots = np.sqrt(np.arange(1, 6))
        signs = np.array([[1 if bits & (1 << j) else -1 for j in range(5)]
                          for bits in range(32)])
        np.testing.assert_allclose(qing(signs * roots), 0, atol=1e-26)
        self.assertEqual(float(qing(np.zeros(5))), 55.)
        self.assertEqual(float(qing(np.ones(5))), 30.)

    def test_reflection_handles_multiple_overshoots(self):
        x = np.array([-2501., -501., -500., 0., 500., 501., 2501.])
        np.testing.assert_allclose(reflect(x, -500., 500.),
                                   [-499., -499., -500., 0., 500., 499., 499.])

    def test_blx_extended_interval(self):
        first = np.zeros((1000, 5))
        second = np.full_like(first, 2.)
        children = blx_alpha(first, second, .5, 1., np.random.default_rng(5))
        self.assertTrue(np.all((children >= -1) & (children <= 3)))
        self.assertTrue(np.any(children < 0) and np.any(children > 2))

    def test_budget_bounds_and_best_so_far(self):
        for algorithm in (genetic_algorithm, random_search):
            for budget in (40, 80, 837):
                parameters = dict(self.parameters, budget=budget)
                result = algorithm(self.problem, parameters, 42)
                self.assertEqual(result.evaluations, budget)
                self.assertEqual(result.history[-1][1], budget)
                self.assertTrue(np.all(np.diff([h[2] for h in result.history]) <= 0))
                self.assertTrue(np.all(np.abs(result.best_x) <= 500))
                self.assertEqual(result.best_f, float(qing(result.best_x)))

    def test_reproducibility_and_independent_seeds(self):
        for algorithm in (genetic_algorithm, random_search):
            first = algorithm(self.problem, self.parameters, 7)
            repeat = algorithm(self.problem, self.parameters, 7)
            other = algorithm(self.problem, self.parameters, 8)
            self.assertEqual(first.history, repeat.history)
            np.testing.assert_array_equal(first.best_x, repeat.best_x)
            self.assertFalse(np.array_equal(first.best_x, other.best_x))

    def test_count_actual_objective_evaluations(self):
        from unittest.mock import patch
        original = qing
        for algorithm in (genetic_algorithm, random_search):
            count = 0
            def counted(x):
                nonlocal count
                count += len(x)
                return original(x)
            with patch('optimizer.qing', side_effect=counted):
                result = algorithm(self.problem, self.parameters, 0)
            self.assertEqual(count, self.parameters['budget'])
            self.assertEqual(count, result.evaluations)

    def test_invalid_parameters_are_rejected(self):
        for changes in ({'budget': 39}, {'population_size': 29},
                        {'mutation_probability': 1.1}, {'mutation_sigma': -1}):
            with self.assertRaises(ValueError):
                genetic_algorithm(self.problem, dict(self.parameters, **changes), 0)


if __name__ == '__main__':
    unittest.main()
