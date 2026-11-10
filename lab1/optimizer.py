"""Генетический алгоритм и случайный поиск для функции Qing."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class Result:
    best_x: np.ndarray
    best_f: float
    evaluations: int
    seconds: float
    # Поколение, число вычислений, лучшее значение к этому моменту.
    history: list[tuple[int, int, float]]


def qing(x: np.ndarray) -> np.ndarray:
    """Qing function. Last axis is the coordinate axis; also accepts batches."""
    x = np.asarray(x, dtype=float)
    indices = np.arange(1, x.shape[-1] + 1)
    return np.sum((x * x - indices) ** 2, axis=-1)


def reflect(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Mirror any out-of-bounds coordinate, including multiple overshoots."""
    width = upper - lower
    if width <= 0:
        raise ValueError("upper must exceed lower")
    phase = np.remainder(x - lower, 2 * width)
    return lower + np.where(phase <= width, phase, 2 * width - phase)


def tournament(population, values, size, count, rng):
    candidates = rng.integers(0, len(population), size=(count, size))
    winners = candidates[np.arange(count), np.argmin(values[candidates], axis=1)]
    return population[winners].copy()


def blx_alpha(first, second, alpha, probability, rng):
    """One child per parent pair; independently sample each extended interval."""
    distance = np.abs(first - second)
    low = np.minimum(first, second) - alpha * distance
    high = np.maximum(first, second) + alpha * distance
    crossed = rng.uniform(low, high)
    use_crossover = rng.random((len(first), 1)) < probability
    return np.where(use_crossover, crossed, first)


def validate(problem, parameters):
    dimension, lower, upper = problem["dimension"], problem["lower"], problem["upper"]
    population, budget = parameters["population_size"], parameters["budget"]
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("bounds must be finite and lower < upper")
    if not isinstance(population, int) or population < 30:
        raise ValueError("population_size must be an integer >= 30")
    if not isinstance(budget, int) or budget < population:
        raise ValueError("budget must be an integer >= population_size")
    if (
        not isinstance(parameters["tournament_size"], int)
        or parameters["tournament_size"] < 1
    ):
        raise ValueError("tournament_size must be a positive integer")
    for key in ("mutation_probability", "crossover_probability"):
        if not 0 <= parameters[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    for key in ("mutation_sigma", "blx_alpha"):
        if not np.isfinite(parameters[key]) or parameters[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")


def genetic_algorithm(problem, parameters, seed):
    validate(problem, parameters)
    start = perf_counter()
    rng = np.random.default_rng(seed)
    dimension = problem["dimension"]
    lower, upper = problem["lower"], problem["upper"]
    population_size, budget = parameters["population_size"], parameters["budget"]
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    values = qing(population)
    best_index = int(np.argmin(values))
    best_x, best_f = population[best_index].copy(), float(values[best_index])
    used, generation = population_size, 0
    history = [(generation, used, best_f)]
    while used < budget:
        count = min(population_size, budget - used)
        first = tournament(
            population, values, parameters["tournament_size"], count, rng
        )
        second = tournament(
            population, values, parameters["tournament_size"], count, rng
        )
        children = blx_alpha(
            first,
            second,
            parameters["blx_alpha"],
            parameters["crossover_probability"],
            rng,
        )
        mask = rng.random(children.shape) < parameters["mutation_probability"]
        children += mask * rng.normal(0, parameters["mutation_sigma"], children.shape)
        children = reflect(children, lower, upper)
        child_values = qing(children)
        used += count
        candidate = int(np.argmin(child_values))
        if child_values[candidate] < best_f:
            best_f = float(child_values[candidate])
            best_x = children[candidate].copy()
        if count == population_size:
            # Возвращаем элиту с уже вычисленным значением функции.
            worst = int(np.argmax(child_values))
            children[worst], child_values[worst] = best_x, best_f
            population, values = children, child_values
        # Неполное последнее поколение учитываем только в результате поиска.
        generation += 1
        history.append((generation, used, best_f))
    return Result(best_x, best_f, used, perf_counter() - start, history)


def random_search(problem, parameters, seed):
    validate(problem, parameters)
    start = perf_counter()
    rng = np.random.default_rng(seed)
    used, batch = 0, 0
    best_f, best_x = float("inf"), None
    history = []
    while used < parameters["budget"]:
        count = min(parameters["population_size"], parameters["budget"] - used)
        points = rng.uniform(
            problem["lower"], problem["upper"], (count, problem["dimension"])
        )
        values = qing(points)
        index = int(np.argmin(values))
        if values[index] < best_f:
            best_x, best_f = points[index].copy(), float(values[index])
        used += count
        history.append((batch, used, best_f))
        batch += 1
    return Result(best_x, best_f, used, perf_counter() - start, history)
