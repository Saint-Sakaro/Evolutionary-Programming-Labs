"""Проверка сохранённых маршрутов, статистики и повторяемости."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-9), (a, b)


def audit(output, repeat=False):
    config = json.loads((output / "config.json").read_text())
    data = json.loads((output / "instance.json").read_text())
    runs = read_csv(output / "runs.csv")
    summary = read_csv(output / "summary.csv")
    history = read_csv(output / "history.csv")
    coordinates, travel, service = (
        data["coordinates_km"],
        data["travel_time_min"],
        data["service_min"],
    )
    n, limit = len(coordinates) - 1, data["time_limit_min"]

    def scalar(route):
        assert sorted(route) == list(range(1, n + 1))
        full = [0] + route + [0]
        length = math.fsum(
            math.dist(coordinates[a], coordinates[b]) for a, b in zip(full, full[1:])
        )
        duration = math.fsum(travel[a][b] for a, b in zip(full, full[1:])) + math.fsum(
            service
        )
        return length, duration, max(duration - limit, 0)

    assert len(runs) == 5 * len(config["seeds"])
    assert len(summary) == 5
    grouped = {}
    for h in history:
        grouped.setdefault((h["method"], int(h["seed"])), []).append(h)
    for r in runs:
        length, duration, violation = scalar(list(map(int, r["route"].split())))
        close(length, r["distance_km"])
        close(duration, r["time_min"])
        close(violation, r["violation_min"])
        assert bool(int(r["feasible"])) == (violation <= 1e-9)
        assert int(r["evaluations"]) == config["common"]["budget"]
        h = grouped[(r["method"], int(r["seed"]))]
        assert [int(v["evaluations"]) for v in h] == list(
            range(
                config["common"]["population_size"],
                config["common"]["budget"],
                config["common"]["population_size"],
            )
        ) + [config["common"]["budget"]]
        assert all(
            float(a["best_fitness"]) >= float(b["best_fitness"])
            for a, b in zip(h, h[1:])
        )
        feasible = [
            float(v["best_feasible_distance"]) for v in h if v["best_feasible_distance"]
        ]
        assert all(a >= b for a, b in zip(feasible, feasible[1:]))
        if int(r["feasible"]):
            close(feasible[-1], r["distance_km"])
        else:
            assert not feasible
    for s in summary:
        method_runs = [r for r in runs if r["method"] == s["method"]]
        assert sorted(int(r["seed"]) for r in method_runs) == sorted(config["seeds"])
        valid = [float(r["distance_km"]) for r in method_runs if int(r["feasible"])]
        assert int(s["feasible_runs"]) == len(valid)
        close(s["success_rate"], len(valid) / len(method_runs))
        if valid:
            for key, value in dict(
                best=min(valid),
                mean=statistics.mean(valid),
                median=statistics.median(valid),
                worst=max(valid),
            ).items():
                close(s[key], value)
        if len(valid) > 1:
            close(s["std"], statistics.stdev(valid))
        else:
            assert s["std"] == ""
        assert int(s["infeasible_elite_runs"]) == sum(
            float(r["elite_violation_min"]) > 1e-9 for r in method_runs
        )
    examples = json.loads((output / "examples.json").read_text())
    assert sum(e["feasible"] for e in examples) == 2 and len(examples) == 4
    for e in examples:
        length, duration, violation = scalar(e["route"])
        close(length, e["distance_km"])
        close(duration, e["time_min"])
        assert e["feasible"] == (violation <= 1e-9)
    schedule = read_csv(output / "best_route.csv")
    assert len(schedule) == n + 2
    assert schedule[0]["node"] == schedule[-1]["node"] == "0"
    route = [int(s["node"]) for s in schedule[1:-1]]
    length, duration, violation = scalar(route)
    assert violation <= 1e-9
    close(length, min(float(r["distance_km"]) for r in runs if int(r["feasible"])))
    close(length, math.fsum(float(s["leg_km"]) for s in schedule))
    close(duration, schedule[-1]["arrival_min"])
    previous = 0.0
    for a, b in zip(schedule, schedule[1:]):
        source, target = int(a["node"]), int(b["node"])
        close(b["travel_min"], travel[source][target])
        close(b["arrival_min"], previous + travel[source][target])
        close(b["service_min"], service[target])
        previous = float(b["arrival_min"]) + service[target]
        close(b["departure_min"], previous)
    manifest = json.loads((output / "manifest.json").read_text())
    for key, filename in [
        ("config_sha256", "config.json"),
        ("instance_sha256", "instance.json"),
    ]:
        assert (
            hashlib.sha256((output / filename).read_bytes()).hexdigest()
            == manifest[key]
        )
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for document in [output / "report.md", ROOT / "README.md"]:
        text = document.read_text()
        assert not any(ord(c) < 32 and c not in "\n\t" for c in text)
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            assert (document.parent / link).exists(), (document, link)
    if repeat:
        from routing import Problem, constructive_search, genetic_algorithm

        problem = Problem.from_dict(data)
        for r in runs:
            name = r["method"]
            if name == "randomized_greedy":
                parameters = dict(config["common"], mutation="swap", penalty=1.0)
                result = constructive_search(
                    problem,
                    parameters,
                    int(r["seed"]),
                    k=config["baseline_candidate_list"],
                )
            else:
                _, mutation, penalty = name.split("_")
                parameters = dict(
                    config["common"], mutation=mutation, penalty=float(penalty[1:])
                )
                result = genetic_algorithm(problem, parameters, int(r["seed"]))
            assert result["feasible"] == bool(int(r["feasible"]))
            assert result["selected"]["distance"] == float(r["distance_km"])
            assert result["selected"]["duration"] == float(r["time_min"])
            assert result["selected"]["route"].tolist() == list(
                map(int, r["route"].split())
            )
            saved = grouped[(name, int(r["seed"]))]
            assert len(saved) == len(result["history"])
            for old, new in zip(saved, result["history"]):
                for key, value in new.items():
                    if value is None:
                        assert old[key] == ""
                    else:
                        assert float(old[key]) == value, (name, key)
        print(
            f"Exact repetition passed for all {len(runs)} runs and {len(history)} history rows."
        )
    print(
        f"Independent scalar audit passed: {len(runs)} routes, statistics, examples, schedule, budgets and hashes."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    audit(args.output, args.repeat)
