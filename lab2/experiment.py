"""Run all four GA configurations and the randomized greedy baseline."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from routing import Problem, evaluate, genetic_algorithm, constructive_search, validate

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config, problem):
    if config['student'] != dict(group=527, number=3, year=2026, variant=12):
        raise ValueError('this report is for group 527, number 3, variant 12')
    seeds = config['seeds']
    if any(type(s) is not int or s < 0 for s in seeds) or len(seeds) < 20 or len(set(seeds)) != len(seeds):
        raise ValueError('provide at least 20 distinct nonnegative integer seeds')
    if sorted(config['mutations']) != ['inversion', 'swap']:
        raise ValueError('compare exactly swap and inversion')
    if len(config['penalties']) != 2 or len(set(config['penalties'])) != 2:
        raise ValueError('provide two distinct penalty coefficients')
    if type(config['baseline_candidate_list']) is not int or not 1 <= config['baseline_candidate_list'] <= problem.n:
        raise ValueError('invalid baseline candidate-list size')
    for mutation in config['mutations']:
        for penalty in config['penalties']:
            validate(dict(config['common'], mutation=mutation, penalty=penalty))


def examples(problem, data):
    first = np.array(data['feasible_witness'], dtype=int)
    valid, invalid = [first], []
    for i in range(problem.n):
        for j in range(i + 1, problem.n):
            other = first.copy()
            other[i], other[j] = other[j], other[i]
            if evaluate(problem, other[None])[2][0] <= 1e-9:
                valid.append(other)
                break
        if len(valid) == 2:
            break
    rng = np.random.default_rng(120527)
    for _ in range(10000):
        other = rng.permutation(np.arange(1, problem.n + 1))
        if evaluate(problem, other[None])[2][0] > 1e-9:
            invalid.append(other)
        if len(invalid) == 2:
            break
    if len(invalid) < 2:
        raise ValueError('could not find two time-infeasible demonstration routes')
    if len(valid) < 2:
        raise ValueError('the instance needs two distinct feasible demonstration routes')
    output = []
    for label, routes in [('feasible', valid), ('infeasible', invalid)]:
        for i, route in enumerate(routes, 1):
            d, t, v = evaluate(problem, route[None])
            output.append(dict(name=f'{label}_{i}', route=route.tolist(), distance_km=float(d[0]),
                               time_min=float(t[0]), violation_min=float(v[0]),
                               feasible=bool(v[0] <= 1e-9)))
    return output


def run(config_path, data_path, output):
    config_bytes, data_bytes = config_path.read_bytes(), data_path.read_bytes()
    config, data = json.loads(config_bytes), json.loads(data_bytes)
    problem = Problem.from_dict(data)
    validate_config(config, problem)
    # Check demo data before creating results. These evaluations are not search evaluations.
    example_routes = examples(problem, data)
    output.mkdir(parents=True, exist_ok=True)
    methods = []
    for mutation in config['mutations']:
        for penalty in config['penalties']:
            name = f'ga_{mutation}_p{penalty:g}'
            methods.append((name, genetic_algorithm, dict(config['common'], mutation=mutation, penalty=penalty)))
    methods.append(('randomized_greedy', constructive_search,
                    dict(config['common'], mutation='swap', penalty=1.0)))
    rows, history, summary = [], [], []
    for name, algorithm, parameters in methods:
        current = []
        for seed in config['seeds']:
            if name == 'randomized_greedy':
                result = algorithm(problem, parameters, seed, k=config['baseline_candidate_list'])
            else:
                result = algorithm(problem, parameters, seed)
            selected, elite = result['selected'], result['penalized_elite']
            row = dict(method=name, seed=seed, feasible=int(result['feasible']),
                       distance_km=selected['distance'], time_min=selected['duration'],
                       violation_min=selected['violation'],
                       route=' '.join(map(str, selected['route'])),
                       evaluations=result['evaluations'], seconds=result['seconds'],
                       feasible_evaluations=result['feasible_evaluations'],
                       elite_distance_km=elite['distance'], elite_time_min=elite['duration'],
                       elite_violation_min=elite['violation'], elite_fitness=elite['fitness'])
            current.append(row)
            rows.append(row)
            history.extend(dict(method=name, seed=seed, **h) for h in result['history'])
        valid = [r for r in current if r['feasible']]
        distances = np.array([r['distance_km'] for r in valid])
        stats = dict(method=name, runs=len(current), feasible_runs=len(valid),
                     success_rate=len(valid) / len(current),
                     best=float(distances.min()) if len(valid) else None,
                     mean=float(distances.mean()) if len(valid) else None,
                     median=float(np.median(distances)) if len(valid) else None,
                     std=float(distances.std(ddof=1)) if len(valid) > 1 else None,
                     worst=float(distances.max()) if len(valid) else None,
                     mean_seconds=float(np.mean([r['seconds'] for r in current])),
                     evaluated_feasible_fraction=sum(r['feasible_evaluations'] for r in current) /
                                                 sum(r['evaluations'] for r in current),
                     infeasible_elite_runs=sum(r['elite_violation_min'] > 1e-9 for r in current),
                     evaluations=parameters['budget'])
        summary.append(stats)
        print(f"{name}: feasible {len(valid)}/{len(current)}, median={stats['median']}, "
              f"infeasible scalar elites={stats['infeasible_elite_runs']}", flush=True)
    write_csv(output / 'runs.csv', rows)
    write_csv(output / 'history.csv', history)
    write_csv(output / 'summary.csv', summary)
    (output / 'config.json').write_bytes(config_bytes)
    (output / 'instance.json').write_bytes(data_bytes)
    (output / 'examples.json').write_text(json.dumps(example_routes, ensure_ascii=False, indent=2) + '\n',
                                          encoding='utf-8')
    feasible_rows = [r for r in rows if r['feasible']]
    if not feasible_rows:
        raise RuntimeError('no feasible routes found; inspect CSV and revise the search budget')
    best = min(feasible_rows, key=lambda r: r['distance_km'])
    route = np.array(list(map(int, best['route'].split())))
    full = [0] + route.tolist() + [0]
    time = 0.
    schedule = [dict(visit=0, node=0, x_km=problem.coordinates[0, 0],
                     y_km=problem.coordinates[0, 1], leg_km=0., travel_min=0.,
                     arrival_min=0., service_min=0., departure_min=0.)]
    for step, (a, b) in enumerate(zip(full, full[1:]), 1):
        arrival = time + problem.travel_time[a, b]
        time = arrival + problem.service[b]
        schedule.append(dict(visit=step, node=b, x_km=problem.coordinates[b, 0],
                             y_km=problem.coordinates[b, 1], leg_km=problem.distance[a, b],
                             travel_min=problem.travel_time[a, b], arrival_min=arrival,
                             service_min=problem.service[b], departure_min=time))
    write_csv(output / 'best_route.csv', schedule)
    import matplotlib
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), numpy=np.__version__,
                    matplotlib=matplotlib.__version__, platform=platform.platform(),
                    rng='numpy.random.default_rng / PCG64',
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    instance_sha256=hashlib.sha256(data_bytes).hexdigest(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(ROOT.glob('*.py'))})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                          encoding='utf-8')
    from plotting import make_plots
    from report import make_report
    make_plots(output, config, problem, rows, history, best)
    make_report(output, config, data, rows, summary, example_routes, best, schedule)
    print(f'Report and data: {output.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'instance.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    run(args.config, args.data, args.output)


if __name__ == '__main__':
    main()
