"""Запуск эксперимента и сохранение результатов."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from optimizer import genetic_algorithm, random_search, validate

ROOT = Path(__file__).resolve().parent


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def validate_config(config):
    # Параметры задачи должны соответствовать варианту в отчёте.
    if config["student"] != {"group": 527, "number": 3, "year": 2026, "variant": 12}:
        raise ValueError(
            "this experiment is for group 527, number 3, variant 12 (2026)"
        )
    if config["problem"] != {"dimension": 5, "lower": -500.0, "upper": 500.0}:
        raise ValueError("variant 12 requires dimension=5 and bounds [-500, 500]")
    seeds = config["seeds"]
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(seeds) < 20 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least 20 distinct seeds")
    configurations = config["configurations"]
    if any(
        not isinstance(c.get("name"), str)
        or not c["name"].strip()
        or c["name"] == "random_search"
        for c in configurations
    ):
        raise ValueError("GA names must be nonempty strings other than random_search")
    if len(configurations) != 2 or len({c["name"] for c in configurations}) != 2:
        raise ValueError("provide two uniquely named GA configurations")
    if any(set(c) != {"name", "mutation_sigma"} for c in configurations):
        raise ValueError("only mutation_sigma may differ between configurations")
    if configurations[0]["mutation_sigma"] == configurations[1]["mutation_sigma"]:
        raise ValueError("the mutation scales must differ")
    threshold = config["success_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("success_threshold must be positive and finite")
    for c in configurations:
        validate(
            config["problem"],
            dict(config["common"], mutation_sigma=c["mutation_sigma"]),
        )


def run(config_path, output):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    seeds, configurations = config["seeds"], config["configurations"]
    methods = [
        (
            c["name"],
            genetic_algorithm,
            dict(config["common"], mutation_sigma=c["mutation_sigma"]),
        )
        for c in configurations
    ]
    methods.append(("random_search", random_search, methods[0][2]))
    output.mkdir(parents=True, exist_ok=True)
    rows, histories, summary = [], [], []
    threshold = config["success_threshold"]
    for name, algorithm, parameters in methods:
        current = []
        for seed in seeds:
            result = algorithm(config["problem"], parameters, seed)
            row = dict(
                method=name,
                seed=seed,
                best_f=result.best_f,
                evaluations=result.evaluations,
                seconds=result.seconds,
                success=int(result.best_f <= threshold),
            )
            row.update(
                {f"x{i + 1}": float(value) for i, value in enumerate(result.best_x)}
            )
            rows.append(row)
            current.append(row)
            histories.extend(
                dict(method=name, seed=seed, generation=g, evaluations=e, best_f=f)
                for g, e, f in result.history
            )
        values = np.array([r["best_f"] for r in current])
        stats = dict(
            method=name,
            runs=len(current),
            best=float(values.min()),
            mean=float(values.mean()),
            median=float(np.median(values)),
            std=float(values.std(ddof=1)),
            worst=float(values.max()),
            success_rate=float(np.mean(values <= threshold)),
            mean_seconds=float(np.mean([r["seconds"] for r in current])),
            evaluations=parameters["budget"],
        )
        summary.append(stats)
        print(
            f"{name}: mean={stats['mean']:.6g}, median={stats['median']:.6g}, "
            f"best={stats['best']:.6g}, successes={sum(r['success'] for r in current)}/{len(current)}",
            flush=True,
        )
    write_csv(output / "runs.csv", rows)
    write_csv(output / "history.csv", histories)
    write_csv(output / "summary.csv", summary)
    (output / "config.json").write_bytes(config_bytes)
    # Сохраняем версии библиотек для повторного запуска.
    import matplotlib

    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
        platform=platform.platform(),
        rng="numpy.random.default_rng / PCG64",
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.glob("*.py"))
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from plotting import make_plots
    from report import make_report

    make_plots(output, config, rows, histories)
    make_report(output, config, rows, summary)
    print(f"Results and report: {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
