"""Генерация координат, времени проезда и обслуживания."""

import argparse
import json
from pathlib import Path

import numpy as np
from routing import Problem, evaluate

ROOT = Path(__file__).resolve().parent


def generate(seed=5270312):
    rng = np.random.default_rng(seed)
    coordinates = np.vstack(([20.0, 20.0], rng.uniform(0, 40, size=(20, 2))))
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    traffic = rng.uniform(0.8, 1.8, size=(21, 21))
    travel = distance / 0.6 * traffic  # 36 km/h nominal; directed congestion factor
    np.fill_diagonal(travel, 0)
    service = np.r_[0, rng.integers(3, 9, size=20)]
    # Строим опорный маршрут, чтобы задать достижимый лимит времени.
    remaining = set(range(1, 21))
    route, current = [], 0
    while remaining:
        current = min(remaining, key=lambda v: (travel[current, v], v))
        route.append(current)
        remaining.remove(current)

    def duration(order):
        full = [0] + order + [0]
        return sum(travel[a, b] for a, b in zip(full, full[1:])) + service.sum()

    # Проверяем инверсии с полным пересчётом направленного времени проезда.
    while True:
        old = duration(route)
        candidates = [
            route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
            for i in range(20)
            for j in range(i + 1, 20)
        ]
        best = min(candidates, key=duration)
        if duration(best) >= old - 1e-9:
            break
        route = best
    limit = float(np.ceil(duration(route) * 1.10))
    data = dict(
        name="Synthetic delivery, 20 customers and depot",
        generation_seed=seed,
        nominal_speed_km_min=0.6,
        traffic_factor_range=[0.8, 1.8],
        time_limit_multiplier=1.10,
        coordinates_km=coordinates.tolist(),
        service_min=service.tolist(),
        travel_time_min=travel.tolist(),
        time_limit_min=limit,
        feasible_witness=route,
        description="Uniform [0,40]^2 customers; depot (20,20); directed fixed traffic; service 3..8 min.",
    )
    problem = Problem.from_dict(data)
    assert evaluate(problem, np.array([route]))[2][0] == 0
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5270312)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "instance.json")
    args = parser.parse_args()
    data = generate(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {args.output}: 20 customers, time limit {data['time_limit_min']:g} min"
    )


if __name__ == "__main__":
    main()
