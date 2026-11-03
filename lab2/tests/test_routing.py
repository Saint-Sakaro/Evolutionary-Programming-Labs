import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from routing import (Problem, Archive, evaluate, ordered_crossover, mutate,
                     genetic_algorithm, constructive_search, constructive_batch)
from experiment import validate_config
from generate_data import generate


class FixedCuts:
    def __init__(self, left, right):
        self.cuts = np.array([left, right])

    def choice(self, *args, **kwargs):
        return self.cuts


class RoutingTests(unittest.TestCase):
    def setUp(self):
        coordinates = np.column_stack((np.arange(21), np.zeros(21)))
        travel = np.full((21, 21), 10.)
        np.fill_diagonal(travel, 0.)
        travel[0, 1], travel[1, 0] = 3., 7.
        self.data = dict(coordinates_km=coordinates.tolist(), travel_time_min=travel.tolist(),
                         service_min=[0] + [2] * 20, time_limit_min=244)
        self.problem = Problem.from_dict(self.data)
        self.params = dict(population_size=40, budget=123, tournament_size=3,
                           crossover_probability=.9, mutation_probability=.3,
                           mutation='swap', penalty=5.)

    def test_distance_depot_return_service_and_directed_time(self):
        routes = np.array([np.arange(1, 21), np.arange(20, 0, -1)])
        distance, time, violation = evaluate(self.problem, routes)
        np.testing.assert_array_equal(distance, [40., 40.])
        np.testing.assert_array_equal(time, [243., 247.])
        np.testing.assert_array_equal(violation, [0., 3.])

    def test_invalid_permutations_are_rejected(self):
        routes = np.arange(1, 21)[None]
        for value in (0, 21, 2):
            broken = routes.copy()
            broken[0, 0] = value
            with self.assertRaises(ValueError):
                evaluate(self.problem, broken)
        with self.assertRaises(ValueError):
            evaluate(self.problem, routes.astype(float))
        with self.assertRaises(ValueError):
            evaluate(self.problem, routes[:, :-1])

    def test_ox_known_cyclic_order(self):
        first = np.arange(1, 9)
        second = np.array([3, 7, 5, 1, 6, 8, 2, 4])
        child = ordered_crossover(first, second, FixedCuts(2, 5))
        np.testing.assert_array_equal(child, [1, 6, 3, 4, 5, 8, 2, 7])
        np.testing.assert_array_equal(first, np.arange(1, 9))

    def test_operators_preserve_permutations_and_parents(self):
        rng = np.random.default_rng(53)
        for _ in range(500):
            first, second = rng.permutation(20) + 1, rng.permutation(20) + 1
            saved = first.copy()
            child = ordered_crossover(first, second, rng)
            for kind in ('swap', 'inversion'):
                np.testing.assert_array_equal(np.sort(mutate(child, kind, rng)), np.arange(1, 21))
            np.testing.assert_array_equal(first, saved)
            np.testing.assert_array_equal(np.sort(child), np.arange(1, 21))

    def test_mutation_semantics(self):
        route = np.arange(1, 6)
        np.testing.assert_array_equal(mutate(route, 'swap', FixedCuts(1, 3)), [1, 4, 3, 2, 5])
        np.testing.assert_array_equal(mutate(route, 'inversion', FixedCuts(0, 3)), [4, 3, 2, 1, 5])

    def test_archive_never_labels_infeasible_scalar_elite_as_solution(self):
        data = copy.deepcopy(self.data)
        travel = np.ones((21, 21))
        np.fill_diagonal(travel, 0)
        travel[0, 1] = 1000
        data.update(travel_time_min=travel.tolist(), time_limit_min=100)
        problem = Problem.from_dict(data)
        bad = np.arange(1, 21)
        good = np.arange(20, 0, -1)
        good[6], good[8] = good[8], good[6]
        archive = Archive(problem, penalty=.001)
        archive.observe(np.array([bad, good]), 0)
        self.assertGreater(archive.elite['violation'], 0)
        self.assertEqual(archive.feasible['violation'], 0)
        np.testing.assert_array_equal(archive.feasible['route'], good)
        self.assertTrue(archive.finish(0)['feasible'])
        np.testing.assert_array_equal(archive.finish(0)['selected']['route'], good)

    def test_missing_feasible_solution_is_reported_as_failure(self):
        problem = copy.deepcopy(self.problem)
        problem.limit = .1  # service alone takes 40 minutes
        for algorithm in (genetic_algorithm, constructive_search):
            result = algorithm(problem, self.params, 2)
            self.assertFalse(result['feasible'])
            self.assertGreater(result['selected']['violation'], 0)
            self.assertTrue(all(h['best_feasible_distance'] is None for h in result['history']))

    def test_mst_lower_bound_on_square(self):
        from report import mst_bound
        self.assertAlmostEqual(mst_bound([[0, 0], [1, 0], [1, 1], [0, 1]]), 3.)

    def test_budget_actual_evaluations_and_feasible_archive_history(self):
        original = evaluate
        for algorithm in (genetic_algorithm, constructive_search):
            for budget in (40, 80, 123):
                count = 0
                def counted(problem, routes):
                    nonlocal count
                    count += len(routes)
                    return original(problem, routes)
                with patch('routing.evaluate', side_effect=counted):
                    result = algorithm(self.problem, dict(self.params, budget=budget), 5)
                self.assertEqual(count, budget)
                self.assertEqual(result['evaluations'], budget)
                history = result['history']
                self.assertTrue(np.all(np.diff([h['best_fitness'] for h in history]) <= 0))
                finite = [h['best_feasible_distance'] for h in history if h['feasible_found']]
                self.assertTrue(np.all(np.diff(finite) <= 0))
                self.assertEqual(history[-1]['evaluations'], budget)

    def test_reproducibility(self):
        for algorithm in (genetic_algorithm, constructive_search):
            a = algorithm(self.problem, self.params, 9)
            b = algorithm(self.problem, self.params, 9)
            self.assertEqual(a['history'], b['history'])
            np.testing.assert_array_equal(a['selected']['route'], b['selected']['route'])

    def test_constructive_batch_visits_all_customers(self):
        for k in (1, 3, 20):
            batch = constructive_batch(self.problem, 100, k, np.random.default_rng(0))
            np.testing.assert_array_equal(np.sort(batch, axis=1),
                                         np.tile(np.arange(1, 21), (100, 1)))

    def test_data_generation_reproduces_saved_instance(self):
        saved = json.loads((ROOT / 'data' / 'instance.json').read_text())
        self.assertEqual(generate(saved['generation_seed']), saved)
        witness = np.array(saved['feasible_witness'])[None]
        self.assertEqual(evaluate(Problem.from_dict(saved), witness)[2][0], 0)

    def test_configuration_validation(self):
        config = json.loads((ROOT / 'config.json').read_text())
        validate_config(config, self.problem)
        for bad in (-1, True, .1, 1):
            changed = copy.deepcopy(config)
            changed['seeds'][0] = bad
            with self.assertRaises(ValueError):
                validate_config(changed, self.problem)


if __name__ == '__main__':
    unittest.main()
