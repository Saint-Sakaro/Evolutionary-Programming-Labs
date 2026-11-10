"""Сравнение методов составления расписания при равном бюджете."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scheduling import (
    Problem,
    evaluate,
    hypervolume,
    nsga2,
    validate,
    weighted_portfolio,
)

ROOT = Path(__file__).resolve().parent


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def coverage(first, second):
    """Fraction of second's vectors weakly dominated by at least one first vector."""
    return float(np.mean(np.all(first[:, None] <= second[None, :], axis=2).any(axis=0)))


def validate_config(config):
    if config["student"] != dict(group=527, number=3, year=2026, variant=12, track="A"):
        raise ValueError(
            "this experiment is for variant 12, track A, group 527, number 3"
        )
    seeds = config["seeds"]
    if (
        len(seeds) < 20
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("provide at least 20 distinct nonnegative integer seeds")
    validate(config["common"])
    weights = np.asarray(config["weights"], dtype=float)
    if weights.ndim != 2 or weights.shape[1] != 3 or len(weights) < 3:
        raise ValueError("provide at least three weight triples")
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(axis=1), 1)
    ):
        raise ValueError("weights must be finite, nonnegative and sum to 1")
    if config["common"]["budget"] // len(weights) < config["common"]["population_size"]:
        raise ValueError("insufficient total budget for the weighted portfolio")


def representative_indices(costs, scale):
    if len(costs) < 3:
        raise ValueError("need at least three distinct nondominated tradeoffs")
    used, result = set(), []
    roles = [
        ("requests", "Соблюдение пожеланий", (0, 1, 2)),
        ("compact", "Компактное расписание", (1, 0, 2)),
        ("early", "Минимум поздних пар", (2, 0, 1)),
    ]
    for key, label, priority in roles:
        order = sorted(
            range(len(costs)), key=lambda i: tuple(costs[i, j] for j in priority)
        )
        selected = next(i for i in order if i not in used)
        used.add(selected)
        result.append((key, label, selected))
    if len(costs) > 3:
        order = np.argsort(np.sum((costs / scale) ** 2, axis=1), kind="stable")
        selected = next(int(i) for i in order if int(i) not in used)
        result.append(("balanced", "Компромиссное расписание", selected))
    return result


def timetable_records(data, assigned, rooms, key):
    result = []
    for e, slot, room in zip(data["events"], assigned, rooms):
        day, period = divmod(int(slot), data["periods"])
        result.append(
            dict(
                solution=key,
                event=e["id"],
                group=data["groups"][e["group"]],
                teacher=data["teachers"][e["teacher"]],
                title=e["title"],
                day=day,
                day_name=data["day_names"][day],
                period=period,
                start=data["period_starts"][period],
                room=int(room),
                room_name=data["room_names"][room],
                requested_slot=e["requested_slot"],
                assigned_slot=int(slot),
                request_conflict=int(slot != e["requested_slot"]),
                late=int(period >= data["late_from"]),
            )
        )
    return result


def run(config_path, data_path, output):
    config_bytes, data_bytes = config_path.read_bytes(), data_path.read_bytes()
    config, data = json.loads(config_bytes), json.loads(data_bytes)
    validate_config(config)
    problem = Problem.from_dict(data)
    output.mkdir(parents=True, exist_ok=True)
    results, runs, history, fronts = {}, [], [], []
    for method in ("nsga2", "weighted_sum"):
        for seed in config["seeds"]:
            result = (
                nsga2(problem, config["common"], seed)
                if method == "nsga2"
                else weighted_portfolio(
                    problem, config["common"], seed, config["weights"]
                )
            )
            results[(method, seed)] = result
            runs.append(
                dict(
                    method=method,
                    seed=seed,
                    evaluations=result["evaluations"],
                    seconds=result["seconds"],
                    front_size=len(result["costs"]),
                    hypervolume=hypervolume(result["costs"], problem.reference),
                    min_conflicts=int(result["costs"][:, 0].min()),
                    min_windows=int(result["costs"][:, 1].min()),
                    min_late=int(result["costs"][:, 2].min()),
                )
            )
            history.extend(
                dict(method=method, seed=seed, **h) for h in result["history"]
            )
            for i, (genes, costs) in enumerate(zip(result["genes"], result["costs"])):
                fronts.append(
                    dict(
                        method=method,
                        seed=seed,
                        point=i,
                        conflicts=int(costs[0]),
                        windows=int(costs[1]),
                        late=int(costs[2]),
                        genes=" ".join(map(str, genes)),
                    )
                )
        values = [r["hypervolume"] for r in runs if r["method"] == method]
        print(
            f"{method}: {len(values)} runs; mean HV={np.mean(values):.6f}; "
            f"median HV={np.median(values):.6f}",
            flush=True,
        )
    for row in runs:
        seed = row["seed"]
        other = "weighted_sum" if row["method"] == "nsga2" else "nsga2"
        row["coverage_of_other"] = coverage(
            results[(row["method"], seed)]["costs"], results[(other, seed)]["costs"]
        )
    summary = []
    for method in ("nsga2", "weighted_sum"):
        current = [r for r in runs if r["method"] == method]
        hv = np.array([r["hypervolume"] for r in current])
        summary.append(
            dict(
                method=method,
                runs=len(current),
                hv_best=float(hv.max()),
                hv_mean=float(hv.mean()),
                hv_median=float(np.median(hv)),
                hv_std=float(hv.std(ddof=1)),
                hv_worst=float(hv.min()),
                mean_front_size=float(np.mean([r["front_size"] for r in current])),
                mean_coverage_of_other=float(
                    np.mean([r["coverage_of_other"] for r in current])
                ),
                mean_seconds=float(np.mean([r["seconds"] for r in current])),
                evaluations=config["common"]["budget"],
            )
        )
    write_csv(output / "runs.csv", runs)
    write_csv(output / "fronts.csv", fronts)
    write_csv(output / "history.csv", history)
    write_csv(output / "summary.csv", summary)
    (output / "config.json").write_bytes(config_bytes)
    (output / "instance.json").write_bytes(data_bytes)
    nsga_runs = [r for r in runs if r["method"] == "nsga2"]
    median_hv = np.median([r["hypervolume"] for r in nsga_runs])
    chosen = min(
        nsga_runs, key=lambda r: (abs(r["hypervolume"] - median_hv), r["seed"])
    )
    seed = chosen["seed"]
    selected_run = results[("nsga2", seed)]
    representatives, timetables = [], []
    for key, label, index in representative_indices(
        selected_run["costs"], problem.reference - 1
    ):
        genes, cost = selected_run["genes"][index], selected_run["costs"][index]
        checked, assigned, rooms = evaluate(problem, genes[None])
        assert np.array_equal(checked[0], cost)
        schedule = timetable_records(data, assigned[0], rooms[0], key)
        timetables.extend(schedule)
        representatives.append(
            dict(
                key=key,
                label=label,
                seed=seed,
                point=index,
                conflicts=int(cost[0]),
                windows=int(cost[1]),
                late=int(cost[2]),
                genes=genes.tolist(),
                timetable=schedule,
            )
        )
    write_csv(output / "timetables.csv", timetables)
    (output / "representatives.json").write_text(
        json.dumps(representatives, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    import matplotlib

    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
        platform=platform.platform(),
        rng="numpy.random.default_rng / PCG64",
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        instance_sha256=hashlib.sha256(data_bytes).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.glob("*.py"))
        },
        hypervolume_reference=problem.reference.tolist(),
        weighted_normalization=(problem.reference - 1).tolist(),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from plotting import make_plots
    from report import make_report

    make_plots(output, config, data, runs, history, results, representatives, seed)
    make_report(output, config, data, runs, summary, representatives, seed)
    print(f"Report: {output.resolve()}; representative NSGA-II seed={seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "instance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.config, args.data, args.output)
