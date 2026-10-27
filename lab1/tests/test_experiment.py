"""Reject configurations that would silently mislabel or merge experiment data."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import validate_config


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'config.json').read_text())

    def test_supplied_configuration(self):
        validate_config(self.config)

    def test_seeds_must_be_distinct_nonnegative_integers(self):
        for bad in (-1, 0.5, True, '7', 1):
            config = copy.deepcopy(self.config)
            config['seeds'][0] = bad
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_ga_name_cannot_collide_with_random_search(self):
        self.config['configurations'][0]['name'] = 'random_search'
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_report_cannot_describe_another_problem(self):
        self.config['problem']['dimension'] = 6
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_threshold_must_be_positive_finite_number(self):
        for threshold in (0, -1, float('nan'), float('inf'), True, '0.1'):
            self.config['success_threshold'] = threshold
            with self.assertRaises(ValueError):
                validate_config(self.config)


if __name__ == '__main__':
    unittest.main()
