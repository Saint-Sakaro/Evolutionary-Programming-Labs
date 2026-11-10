import copy
import itertools
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import coverage, validate_config
from generate_data import generate
from scheduling import (
    Archive,
    Problem,
    crowding_distance,
    dominance_matrix,
    environmental_selection,
    evaluate,
    hypervolume,
    nondominated_sort,
    nsga2,
    weighted_portfolio,
)


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "data" / "instance.json").read_text())
        self.problem = Problem.from_dict(self.data)
        self.parameters = dict(
            population_size=30,
            budget=97,
            crossover_probability=0.9,
            mutation_probability=1 / 24,
        )

    def test_request_anchor_has_known_objectives(self):
        costs, slots, _ = evaluate(self.problem, self.problem.requested[None])
        np.testing.assert_array_equal(slots[0], self.problem.requested)
        np.testing.assert_array_equal(costs[0], [0, 16, 12])

    def test_decoder_enforces_group_teacher_and_room_exclusion(self):
        rng = np.random.default_rng(83)
        genes = np.vstack(
            (np.zeros((1, 24), dtype=int), rng.integers(30, size=(300, 24)))
        )
        _, slots, rooms = evaluate(self.problem, genes)
        for assignments, room in zip(slots, rooms):
            for i, j in itertools.combinations(range(24), 2):
                if assignments[i] == assignments[j]:
                    self.assertNotEqual(room[i], room[j])
                    self.assertNotEqual(self.problem.groups[i], self.problem.groups[j])
                    self.assertNotEqual(
                        self.problem.teachers[i], self.problem.teachers[j]
                    )

    def test_bad_genes_rejected(self):
        for genes in (
            np.zeros((1, 24)),
            np.full((1, 24), 30),
            np.full((1, 24), -1),
            np.zeros((1, 23), dtype=int),
        ):
            with self.assertRaises(ValueError):
                evaluate(self.problem, genes)

    def test_nondominated_ranks_with_duplicates_and_ties(self):
        values = np.array(
            [[0, 3, 3], [3, 0, 3], [3, 3, 0], [1, 4, 4], [4, 4, 4], [0, 3, 3]]
        )
        fronts, ranks = nondominated_sort(values)
        self.assertEqual([set(f) for f in fronts], [{0, 1, 2, 5}, {3}, {4}])
        np.testing.assert_array_equal(ranks, [0, 0, 0, 1, 2, 0])
        self.assertFalse(np.diag(dominance_matrix(values)).any())

    def test_crowding_values_and_constant_objective(self):
        values = np.array([[0, 2, 5], [1, 1, 5], [2, 0, 5]])
        crowding = crowding_distance(values, np.arange(3))
        self.assertTrue(np.isinf(crowding[[0, 2]]).all())
        self.assertEqual(crowding[1], 2.0)
        np.testing.assert_array_equal(
            crowding_distance(np.ones((4, 3)), np.arange(4)), np.zeros(4)
        )

    def test_environmental_selection_keeps_extreme_points(self):
        genes = np.arange(5)[:, None]
        costs = np.array([[0, 4, 0], [1, 3, 0], [2, 2, 0], [3, 1, 0], [4, 0, 0]])
        selected, _ = environmental_selection(genes, costs, 2)
        self.assertEqual(set(selected.ravel()), {0, 4})

    def test_hypervolume_matches_exhaustive_integer_cells(self):
        reference = np.array([4, 5, 6])
        rng = np.random.default_rng(10)
        for _ in range(20):
            points = rng.integers(reference, size=(10, 3))
            count = sum(
                any(np.all(point <= cell) for point in points)
                for cell in itertools.product(range(4), range(5), range(6))
            )
            self.assertAlmostEqual(hypervolume(points, reference), count / 120)
        self.assertAlmostEqual(hypervolume([[1, 1, 1]], [3, 3, 3]), 8 / 27)

    def test_archive_removes_duplicates_and_dominated_points(self):
        archive = Archive(self.problem)
        genes = np.zeros((4, 24), dtype=int)
        archive.observe(genes, np.array([[3, 3, 3], [2, 2, 2], [2, 2, 2], [1, 4, 1]]))
        self.assertEqual({tuple(v) for v in archive.costs}, {(2, 2, 2), (1, 4, 1)})
        before = archive.history[-1]["hypervolume"]
        archive.observe(genes[:1], np.array([[1, 1, 1]]))
        self.assertGreaterEqual(archive.history[-1]["hypervolume"], before)
        self.assertEqual(archive.evaluations, 5)

    def test_coverage_weak_dominance_definition(self):
        a = np.array([[1, 1, 1]])
        b = np.array([[1, 1, 1], [2, 2, 2], [0, 2, 2]])
        self.assertAlmostEqual(coverage(a, b), 2 / 3)
        self.assertEqual(coverage(b, a), 1.0)

    def test_actual_budget_including_partial_generations_and_weight_restarts(self):
        original = evaluate
        weights = np.eye(3)
        for method in ("nsga2", "weighted_sum"):
            count = 0

            def counted(problem, genes):
                nonlocal count
                count += len(genes)
                return original(problem, genes)

            with patch("scheduling.evaluate", side_effect=counted):
                result = (
                    nsga2(self.problem, self.parameters, 1)
                    if method == "nsga2"
                    else weighted_portfolio(self.problem, self.parameters, 1, weights)
                )
            self.assertEqual(count, 97)
            self.assertEqual(result["evaluations"], 97)
            self.assertEqual(result["history"][-1]["evaluations"], 97)
            self.assertTrue(
                np.all(np.diff([h["hypervolume"] for h in result["history"]]) >= 0)
            )
            self.assertFalse(dominance_matrix(result["costs"]).any())

    def test_reproducible_searches(self):
        for algorithm in (nsga2, weighted_portfolio):
            args = (self.problem, self.parameters, 8)
            kwargs = {"weights": np.eye(3)} if algorithm is weighted_portfolio else {}
            a, b = algorithm(*args, **kwargs), algorithm(*args, **kwargs)
            np.testing.assert_array_equal(a["genes"], b["genes"])
            np.testing.assert_array_equal(a["costs"], b["costs"])
            self.assertEqual(a["history"], b["history"])

    def test_generator_reproduces_instance(self):
        self.assertEqual(generate(self.data["generation_seed"]), self.data)

    def test_configuration_validation(self):
        c = json.loads((ROOT / "config.json").read_text())
        validate_config(c)
        for bad in (-1, True, 0.5, 1):
            other = copy.deepcopy(c)
            other["seeds"][0] = bad
            with self.assertRaises(ValueError):
                validate_config(other)


if __name__ == "__main__":
    unittest.main()
