"""Декодирование расписаний, NSGA-II и метод взвешенной суммы."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class Problem:
    groups: np.ndarray
    teachers: np.ndarray
    requested: np.ndarray
    group_count: int = 4
    teacher_count: int = 4
    days: int = 5
    periods: int = 6
    rooms: int = 2
    late_from: int = 4

    @property
    def n(self):
        return len(self.groups)

    @property
    def slots(self):
        return self.days * self.periods

    @property
    def reference(self):
        # Общая опорная точка для вычисления гиперобъёма.
        return np.array(
            [
                self.n + 1,
                self.group_count * self.days * (self.periods - 2) + 1,
                self.n + 1,
            ],
            dtype=int,
        )

    @classmethod
    def from_dict(cls, data):
        if (data["days"], data["periods"], data["rooms"], data["late_from"]) != (
            5,
            6,
            2,
            4,
        ):
            raise ValueError(
                "this laboratory model requires 5 days, 6 periods, 2 rooms, late_from=4"
            )
        events = data["events"]
        problem = cls(
            np.array([e["group"] for e in events], dtype=int),
            np.array([e["teacher"] for e in events], dtype=int),
            np.array([e["requested_slot"] for e in events], dtype=int),
        )
        if problem.n != 24 or [e["id"] for e in events] != list(range(24)):
            raise ValueError("expected 24 sequentially numbered events")
        if not np.all((problem.groups >= 0) & (problem.groups < 4)) or not np.all(
            (problem.teachers >= 0) & (problem.teachers < 4)
        ):
            raise ValueError("group/teacher IDs must be in [0,3]")
        if not np.all(np.bincount(problem.groups, minlength=4) == 6) or not np.all(
            np.bincount(problem.teachers, minlength=4) == 6
        ):
            raise ValueError("each group and teacher must have exactly six events")
        if not np.all((problem.requested >= 0) & (problem.requested < problem.slots)):
            raise ValueError("invalid requested slot")
        return problem


def evaluate(problem, genes):
    """Decode integer target slots to feasible (time,room); return 3 costs."""
    genes = np.asarray(genes)
    if (
        genes.ndim != 2
        or genes.shape[1] != problem.n
        or not np.issubdtype(genes.dtype, np.integer)
    ):
        raise ValueError("expected a batch of 24 integer target-slot vectors")
    if not np.all((genes >= 0) & (genes < problem.slots)):
        raise ValueError("target slots must be in [0,29]")
    batch_size = len(genes)
    rows = np.arange(batch_size)
    busy_groups = np.zeros((batch_size, problem.group_count, problem.slots), dtype=bool)
    busy_teachers = np.zeros(
        (batch_size, problem.teacher_count, problem.slots), dtype=bool
    )
    room_count = np.zeros((batch_size, problem.slots), dtype=int)
    assigned = np.empty_like(genes)
    rooms = np.empty_like(genes)
    for event_index in range(problem.n):
        candidates = (
            genes[:, event_index, None] + np.arange(problem.slots)
        ) % problem.slots
        group, teacher = problem.groups[event_index], problem.teachers[event_index]
        valid = ~(
            busy_groups[rows[:, None], group, candidates]
            | busy_teachers[rows[:, None], teacher, candidates]
        )
        valid &= room_count[rows[:, None], candidates] < problem.rooms
        if not np.all(valid.any(axis=1)):
            raise ValueError("decoder cannot place an event")
        chosen = candidates[rows, np.argmax(valid, axis=1)]
        assigned[:, event_index] = chosen
        rooms[:, event_index] = room_count[rows, chosen]
        room_count[rows, chosen] += 1
        busy_groups[rows, group, chosen] = True
        busy_teachers[rows, teacher, chosen] = True
    conflicts = (assigned != problem.requested).sum(axis=1)
    windows = np.zeros(batch_size, dtype=int)
    for group in range(problem.group_count):
        for day in range(problem.days):
            occupied = busy_groups[
                :, group, day * problem.periods : (day + 1) * problem.periods
            ]
            count = occupied.sum(axis=1)
            first = np.min(
                np.where(occupied, np.arange(problem.periods), problem.periods), axis=1
            )
            last = np.max(np.where(occupied, np.arange(problem.periods), -1), axis=1)
            windows += np.where(count > 0, last - first + 1 - count, 0)
    late = ((assigned % problem.periods) >= problem.late_from).sum(axis=1)
    return np.column_stack((conflicts, windows, late)), assigned, rooms


def dominance_matrix(costs):
    """Entry [i,j] is true iff i strictly Pareto-dominates j."""
    costs = np.asarray(costs)
    return np.all(costs[:, None, :] <= costs[None, :, :], axis=2) & np.any(
        costs[:, None, :] < costs[None, :, :], axis=2
    )


def nondominated_sort(costs):
    dominates = dominance_matrix(costs)
    counts = dominates.sum(axis=0)
    ranks = np.full(len(costs), -1, dtype=int)
    fronts = []
    while np.any(ranks < 0):
        front = np.flatnonzero((counts == 0) & (ranks < 0))
        if not len(front):
            raise RuntimeError("invalid dominance graph")
        ranks[front] = len(fronts)
        fronts.append(front)
        counts -= dominates[front].sum(axis=0)
    return fronts, ranks


def crowding_distance(costs, front):
    """Normalized neighbor gaps; constant objectives contribute zero."""
    values = np.asarray(costs)[front]
    distance = np.zeros(len(front))
    if len(front) <= 2:
        distance[:] = np.inf
        return distance
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective], kind="stable")
        span = float(values[order[-1], objective] - values[order[0], objective])
        if span == 0:
            continue
        distance[order[0]] = distance[order[-1]] = np.inf
        distance[order[1:-1]] += (
            values[order[2:], objective] - values[order[:-2], objective]
        ) / span
    return distance


def rank_and_crowding(costs):
    fronts, ranks = nondominated_sort(costs)
    crowding = np.zeros(len(costs))
    for front in fronts:
        crowding[front] = crowding_distance(costs, front)
    return fronts, ranks, crowding


def environmental_selection(genes, costs, count):
    fronts, _, crowding = rank_and_crowding(costs)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= count:
            selected.extend(front.tolist())
        else:
            order = np.argsort(-crowding[front], kind="stable")
            selected.extend(front[order[: count - len(selected)]].tolist())
            break
    return genes[selected].copy(), costs[selected].copy()


def hypervolume(costs, reference):
    """Exact 3D dominated union volume, divided by the reference box volume."""
    points = np.unique(np.asarray(costs, dtype=int), axis=0)
    reference = np.asarray(reference, dtype=int)
    if not len(points):
        return 0.0
    if points.shape[1] != 3 or np.any(points < 0) or np.any(points >= reference):
        raise ValueError("objectives must be nonnegative and strictly below reference")
    volume = 0
    xs = np.unique(points[:, 0])
    for index, x in enumerate(xs):
        next_x = xs[index + 1] if index + 1 < len(xs) else reference[0]
        yz = points[points[:, 0] <= x, 1:]
        yz = yz[np.argsort(yz[:, 0], kind="stable")]
        heights = reference[2] - np.minimum.accumulate(yz[:, 1])
        widths = np.diff(np.r_[yz[:, 0], reference[1]])
        area = int(np.sum(widths * heights))
        volume += int(next_x - x) * area
    return volume / int(np.prod(reference))


class Archive:
    """Unique objective vectors, nondominated among all evaluated candidates."""

    def __init__(self, problem):
        self.problem = problem
        self.genes = np.empty((0, problem.n), dtype=int)
        self.costs = np.empty((0, 3), dtype=int)
        self.evaluations = 0
        self.history = []

    def observe(self, genes, costs):
        self.evaluations += len(genes)
        combined_genes = np.vstack((self.genes, genes))
        combined_costs = np.vstack((self.costs, costs))
        _, unique = np.unique(combined_costs, axis=0, return_index=True)
        combined_genes, combined_costs = combined_genes[unique], combined_costs[unique]
        keep = ~dominance_matrix(combined_costs).any(axis=0)
        self.genes, self.costs = combined_genes[keep], combined_costs[keep]
        self.history.append(
            dict(
                evaluations=self.evaluations,
                front_size=len(self.costs),
                hypervolume=hypervolume(self.costs, self.problem.reference),
            )
        )


def validate(parameters):
    if (
        type(parameters["population_size"]) is not int
        or parameters["population_size"] < 30
    ):
        raise ValueError("population_size must be an integer >=30")
    if (
        type(parameters["budget"]) is not int
        or parameters["budget"] < parameters["population_size"]
    ):
        raise ValueError("budget must be an integer >=population_size")
    for key in ("crossover_probability", "mutation_probability"):
        if not 0 <= parameters[key] <= 1:
            raise ValueError(f"invalid {key}")


def initial_population(problem, population_size, rng):
    population = rng.integers(problem.slots, size=(population_size, problem.n))
    # Оба метода начинают с расписания по пожеланиям.
    population[0] = problem.requested
    return population


def offspring(problem, population, winners, parameters, rng):
    count = len(winners) // 2
    first, second = population[winners[0::2]], population[winners[1::2]]
    crossover = rng.random((count, 1)) < parameters["crossover_probability"]
    mask = (rng.random(first.shape) < 0.5) & crossover
    children = np.where(mask, second, first)
    mutation = rng.random(children.shape) < parameters["mutation_probability"]
    replacements = rng.integers(problem.slots, size=children.shape)
    return np.where(mutation, replacements, children)


def nsga2(problem, parameters, seed):
    validate(parameters)
    started = perf_counter()
    rng = np.random.default_rng(seed)
    population_size = parameters["population_size"]
    population = initial_population(problem, population_size, rng)
    costs = evaluate(problem, population)[0]
    archive = Archive(problem)
    archive.observe(population, costs)
    while archive.evaluations < parameters["budget"]:
        count = min(population_size, parameters["budget"] - archive.evaluations)
        _, ranks, crowding = rank_and_crowding(costs)
        candidates = rng.integers(len(population), size=(2 * count, 2))
        first, second = candidates[:, 0], candidates[:, 1]
        take_second = (ranks[second] < ranks[first]) | (
            (ranks[second] == ranks[first]) & (crowding[second] > crowding[first])
        )
        winners = np.where(take_second, second, first)
        children = offspring(problem, population, winners, parameters, rng)
        child_costs = evaluate(problem, children)[0]
        archive.observe(children, child_costs)
        population, costs = environmental_selection(
            np.vstack((population, children)),
            np.vstack((costs, child_costs)),
            population_size,
        )
    return dict(
        genes=archive.genes,
        costs=archive.costs,
        evaluations=archive.evaluations,
        history=archive.history,
        seconds=perf_counter() - started,
    )


def weighted_portfolio(problem, parameters, seed, weights):
    validate(parameters)
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 2
        or weights.shape[1] != 3
        or np.any(weights < 0)
        or not np.isfinite(weights).all()
    ):
        raise ValueError("weights must be finite nonnegative triples")
    if not np.allclose(weights.sum(axis=1), 1):
        raise ValueError("each weight triple must sum to 1")
    population_size = parameters["population_size"]
    budgets = [
        parameters["budget"] // len(weights) + (i < parameters["budget"] % len(weights))
        for i in range(len(weights))
    ]
    if min(budgets) < population_size:
        raise ValueError("total budget must provide one population per weight")
    started = perf_counter()
    rng = np.random.default_rng(seed)
    archive = Archive(problem)
    # Нормируем критерии по их верхним границам.
    scale = problem.reference - 1
    for weight, budget in zip(weights, budgets):
        population = initial_population(problem, population_size, rng)
        costs = evaluate(problem, population)[0]
        archive.observe(population, costs)
        used = population_size
        scores = (costs / scale) @ weight
        elite_index = int(np.argmin(scores))
        elite_gene, elite_cost, elite_score = (
            population[elite_index].copy(),
            costs[elite_index].copy(),
            scores[elite_index],
        )
        while used < budget:
            count = min(population_size, budget - used)
            candidates = rng.integers(population_size, size=(2 * count, 2))
            winners = candidates[
                np.arange(2 * count), np.argmin(scores[candidates], axis=1)
            ]
            children = offspring(problem, population, winners, parameters, rng)
            child_costs = evaluate(problem, children)[0]
            archive.observe(children, child_costs)
            child_scores = (child_costs / scale) @ weight
            used += count
            best = int(np.argmin(child_scores))
            if child_scores[best] < elite_score:
                elite_gene, elite_cost, elite_score = (
                    children[best].copy(),
                    child_costs[best].copy(),
                    child_scores[best],
                )
            if count == population_size:
                worst = int(np.argmax(child_scores))
                children[worst], child_costs[worst], child_scores[worst] = (
                    elite_gene,
                    elite_cost,
                    elite_score,
                )
                population, costs, scores = children, child_costs, child_scores
    return dict(
        genes=archive.genes,
        costs=archive.costs,
        evaluations=archive.evaluations,
        history=archive.history,
        seconds=perf_counter() - started,
    )
