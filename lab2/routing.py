"""Own permutation GA and a randomized constructive baseline for delivery routing."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np


@dataclass
class Problem:
    coordinates: np.ndarray
    distance: np.ndarray
    travel_time: np.ndarray
    service: np.ndarray
    limit: float

    @property
    def n(self):
        return len(self.coordinates) - 1

    @classmethod
    def from_dict(cls, data):
        coordinates = np.asarray(data['coordinates_km'], dtype=float)
        distance = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        problem = cls(coordinates, distance,
                      np.asarray(data['travel_time_min'], dtype=float),
                      np.asarray(data['service_min'], dtype=float),
                      float(data['time_limit_min']))
        m = len(coordinates)
        if coordinates.shape != (m, 2) or not 16 <= m <= 26:
            raise ValueError('use a depot plus 15–25 customers with 2D coordinates')
        if problem.travel_time.shape != (m, m) or problem.service.shape != (m,):
            raise ValueError('invalid time matrix or service vector')
        if not all(np.isfinite(a).all() for a in (coordinates, distance, problem.travel_time, problem.service)):
            raise ValueError('data must be finite')
        if np.any(problem.travel_time < 0) or np.any(problem.service < 0):
            raise ValueError('times must be nonnegative')
        if problem.service[0] != 0 or np.any(np.diag(problem.travel_time) != 0):
            raise ValueError('depot service and diagonal travel times must be zero')
        if not np.isfinite(problem.limit) or problem.limit <= 0:
            raise ValueError('time limit must be positive')
        return problem


def evaluate(problem, routes):
    """Every row visits customers 1..n exactly once; depot 0 is implicit."""
    routes = np.asarray(routes)
    if routes.ndim != 2 or routes.shape[1] != problem.n:
        raise ValueError('expected a batch of n-customer permutations')
    if not np.issubdtype(routes.dtype, np.integer):
        raise ValueError('customer IDs must be integers')
    if not np.all(np.sort(routes, axis=1) == np.arange(1, problem.n + 1)):
        raise ValueError('each route must contain every customer exactly once')
    zeros = np.zeros((len(routes), 1), dtype=int)
    full = np.concatenate((zeros, routes, zeros), axis=1)
    source, target = full[:, :-1], full[:, 1:]
    distance = problem.distance[source, target].sum(axis=1)
    duration = problem.travel_time[source, target].sum(axis=1) + problem.service.sum()
    violation = np.maximum(duration - problem.limit, 0)
    return distance, duration, violation


def ordered_crossover(first, second, rng):
    """OX: preserve [left:right) from first, fill cyclically from second."""
    n = len(first)
    left, right = sorted(rng.choice(n + 1, 2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[left:right] = first[left:right]
    used = np.zeros(n + 1, dtype=bool)
    used[first[left:right]] = True
    cyclic = np.concatenate((second[right:], second[:right]))
    remaining = cyclic[~used[cyclic]]
    positions = np.concatenate((np.arange(right, n), np.arange(left)))
    child[positions] = remaining
    return child


def mutate(route, kind, rng):
    result = route.copy()
    left, right = sorted(rng.choice(len(route), 2, replace=False))
    if kind == 'swap':
        result[left], result[right] = result[right], result[left]
    elif kind == 'inversion':
        result[left:right + 1] = result[left:right + 1][::-1]
    else:
        raise ValueError('mutation must be swap or inversion')
    return result


def validate(parameters):
    for key, minimum in [('population_size', 30), ('tournament_size', 1)]:
        if type(parameters[key]) is not int or parameters[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if type(parameters['budget']) is not int or parameters['budget'] < parameters['population_size']:
        raise ValueError('budget must be an integer >= population_size')
    for key in ('crossover_probability', 'mutation_probability'):
        if not 0 <= parameters[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    if parameters['mutation'] not in ('swap', 'inversion'):
        raise ValueError('unknown mutation')
    if not np.isfinite(parameters['penalty']) or parameters['penalty'] <= 0:
        raise ValueError('penalty must be positive and finite')


class Archive:
    """Keep the scalar-fitness elite AND the best feasible route separately."""

    def __init__(self, problem, penalty):
        self.problem, self.penalty = problem, penalty
        self.elite = None
        self.feasible = None
        self.used = 0
        self.feasible_evaluations = 0
        self.history = []

    def observe(self, routes, generation):
        distance, duration, violation = evaluate(self.problem, routes)
        scores = distance + self.penalty * violation
        self.used += len(routes)
        valid = violation <= 1e-9
        self.feasible_evaluations += int(valid.sum())
        i = int(np.argmin(scores))
        if self.elite is None or scores[i] < self.elite['fitness']:
            self.elite = self.record(routes[i], distance[i], duration[i], violation[i], scores[i])
        indices = np.flatnonzero(valid)
        if len(indices):
            j = int(indices[np.argmin(distance[indices])])
            if self.feasible is None or distance[j] < self.feasible['distance']:
                self.feasible = self.record(routes[j], distance[j], duration[j], violation[j], scores[j])
        self.history.append(dict(generation=generation, evaluations=self.used,
                                 best_fitness=float(self.elite['fitness']),
                                 best_feasible_distance=None if self.feasible is None else self.feasible['distance'],
                                 feasible_found=int(self.feasible is not None),
                                 feasible_evaluations=self.feasible_evaluations))
        return scores

    @staticmethod
    def record(route, distance, duration, violation, fitness):
        return dict(route=route.copy(), distance=float(distance), duration=float(duration),
                    violation=float(violation), fitness=float(fitness))

    def finish(self, started):
        selected = self.feasible if self.feasible is not None else self.elite
        return dict(selected=selected, penalized_elite=self.elite,
                    feasible=self.feasible is not None, evaluations=self.used,
                    feasible_evaluations=self.feasible_evaluations,
                    seconds=perf_counter() - started, history=self.history)


def genetic_algorithm(problem, parameters, seed):
    validate(parameters)
    started = perf_counter()
    rng = np.random.default_rng(seed)
    n = parameters['population_size']
    population = np.array([rng.permutation(np.arange(1, problem.n + 1)) for _ in range(n)])
    archive = Archive(problem, parameters['penalty'])
    scores = archive.observe(population, 0)
    generation = 0
    while archive.used < parameters['budget']:
        count = min(n, parameters['budget'] - archive.used)
        contenders = rng.integers(n, size=(2 * count, parameters['tournament_size']))
        winners = contenders[np.arange(2 * count), np.argmin(scores[contenders], axis=1)]
        children = []
        for index in range(count):
            a, b = population[winners[2 * index]], population[winners[2 * index + 1]]
            child = ordered_crossover(a, b, rng) if rng.random() < parameters['crossover_probability'] else a.copy()
            if rng.random() < parameters['mutation_probability']:
                child = mutate(child, parameters['mutation'], rng)
            children.append(child)
        children = np.asarray(children)
        generation += 1
        child_scores = archive.observe(children, generation)
        if count == n:
            worst = int(np.argmax(child_scores))
            children[worst] = archive.elite['route']
            child_scores[worst] = archive.elite['fitness']
            population, scores = children, child_scores
    return archive.finish(started)


def constructive_batch(problem, count, k, rng):
    """Randomized nearest-neighbor by directed travel time, restricted list size k."""
    routes = np.empty((count, problem.n), dtype=int)
    visited = np.zeros((count, problem.n + 1), dtype=bool)
    visited[:, 0] = True
    current = np.zeros(count, dtype=int)
    rows = np.arange(count)
    for step in range(problem.n):
        costs = problem.travel_time[current].copy()
        costs[visited] = np.inf
        width = min(k, problem.n - step)
        nearest = np.argsort(costs, axis=1, kind='stable')[:, :width]
        chosen = nearest[rows, rng.integers(width, size=count)]
        routes[:, step] = chosen
        visited[rows, chosen] = True
        current = chosen
    return routes


def constructive_search(problem, parameters, seed, k=3):
    validate(parameters)
    if type(k) is not int or not 1 <= k <= problem.n:
        raise ValueError('candidate-list size must be in [1,n]')
    started = perf_counter()
    rng = np.random.default_rng(seed)
    archive = Archive(problem, parameters['penalty'])
    batch = 0
    while archive.used < parameters['budget']:
        count = min(parameters['population_size'], parameters['budget'] - archive.used)
        archive.observe(constructive_batch(problem, count, k, rng), batch)
        batch += 1
    return archive.finish(started)
