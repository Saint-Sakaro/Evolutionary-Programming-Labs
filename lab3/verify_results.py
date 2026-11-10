"""Независимая проверка расписаний, гиперобъёма и повторяемости."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scalar_decode(data, genes):
    """Декодируем через множества и циклы независимо от основного алгоритма."""
    assert len(genes) == 24 and all(type(g) is int and 0 <= g < 30 for g in genes)
    busy_groups, busy_teachers, busy_rooms = set(), set(), set()
    assigned, rooms = [], []
    for e, target in zip(data["events"], genes):
        for offset in range(30):
            slot = (target + offset) % 30
            if (e["group"], slot) in busy_groups or (
                e["teacher"],
                slot,
            ) in busy_teachers:
                continue
            available = [r for r in range(2) if (r, slot) not in busy_rooms]
            if not available:
                continue
            room = available[0]
            assigned.append(slot)
            rooms.append(room)
            busy_groups.add((e["group"], slot))
            busy_teachers.add((e["teacher"], slot))
            busy_rooms.add((room, slot))
            break
        else:
            raise AssertionError("scalar decoder failed")
    conflicts = sum(s != e["requested_slot"] for s, e in zip(assigned, data["events"]))
    windows = 0
    for group in range(4):
        for day in range(5):
            periods = [
                s % 6
                for s, e in zip(assigned, data["events"])
                if e["group"] == group and s // 6 == day
            ]
            if periods:
                windows += max(periods) - min(periods) + 1 - len(periods)
    late = sum(s % 6 >= 4 for s in assigned)
    return [conflicts, windows, late], assigned, rooms


def cell_hypervolume(costs, reference):
    occupied = np.zeros(reference, dtype=bool)
    for a, b, c in costs:
        occupied[a:, b:, c:] = True
    return float(occupied.mean())


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def audit(output, repeat=False):
    config = json.loads((output / "config.json").read_text())
    data = json.loads((output / "instance.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    reference = manifest["hypervolume_reference"]
    assert reference == [25, 81, 25]
    fronts = read_csv(output / "fronts.csv")
    runs = read_csv(output / "runs.csv")
    summary = read_csv(output / "summary.csv")
    history = read_csv(output / "history.csv")
    grouped = {}
    for row in fronts:
        genes = list(map(int, row["genes"].split()))
        costs, _, _ = scalar_decode(data, genes)
        assert costs == [int(row[k]) for k in ("conflicts", "windows", "late")]
        grouped.setdefault((row["method"], int(row["seed"])), []).append(row)
    histories = {}
    for h in history:
        histories.setdefault((h["method"], int(h["seed"])), []).append(h)
    assert len(runs) == 2 * len(config["seeds"])

    def vectors(key):
        return [
            [int(r[k]) for k in ("conflicts", "windows", "late")] for r in grouped[key]
        ]

    for r in runs:
        key = (r["method"], int(r["seed"]))
        costs = vectors(key)
        assert len(costs) == int(r["front_size"])
        assert len(set(tuple(c) for c in costs)) == len(costs)
        for i, a in enumerate(costs):
            for j, b in enumerate(costs):
                assert not (
                    i != j
                    and all(x <= y for x, y in zip(a, b))
                    and any(x < y for x, y in zip(a, b))
                )
        close(r["hypervolume"], cell_hypervolume(costs, reference))
        assert int(r["evaluations"]) == config["common"]["budget"]
        h = histories[key]
        assert int(h[-1]["evaluations"]) == config["common"]["budget"]
        assert all(
            int(a["evaluations"]) < int(b["evaluations"]) for a, b in zip(h, h[1:])
        )
        assert all(
            float(a["hypervolume"]) <= float(b["hypervolume"]) for a, b in zip(h, h[1:])
        )
        close(h[-1]["hypervolume"], r["hypervolume"])
        assert int(h[-1]["front_size"]) == len(costs)
        other = "weighted_sum" if key[0] == "nsga2" else "nsga2"
        other_costs = vectors((other, key[1]))
        covered = sum(
            any(all(x <= y for x, y in zip(a, b)) for a in costs) for b in other_costs
        )
        close(r["coverage_of_other"], covered / len(other_costs))
    for s in summary:
        current = [r for r in runs if r["method"] == s["method"]]
        assert sorted(int(r["seed"]) for r in current) == sorted(config["seeds"])
        hv = [float(r["hypervolume"]) for r in current]
        for key, value in dict(
            hv_best=max(hv),
            hv_mean=statistics.mean(hv),
            hv_median=statistics.median(hv),
            hv_std=statistics.stdev(hv),
            hv_worst=min(hv),
        ).items():
            close(s[key], value)
    representatives = json.loads((output / "representatives.json").read_text())
    assert len(representatives) >= 3
    assert len(
        {(r["conflicts"], r["windows"], r["late"]) for r in representatives}
    ) == len(representatives)
    timetable_csv = read_csv(output / "timetables.csv")
    for rep in representatives:
        costs, assigned, rooms = scalar_decode(data, rep["genes"])
        assert costs == [rep["conflicts"], rep["windows"], rep["late"]]
        assert len(rep["timetable"]) == 24
        for e, item, slot, room in zip(
            data["events"], rep["timetable"], assigned, rooms
        ):
            assert item["event"] == e["id"]
            assert item["group"] == data["groups"][e["group"]]
            assert item["teacher"] == data["teachers"][e["teacher"]]
            assert (item["assigned_slot"], item["room"]) == (slot, room)
            assert item["title"] == e["title"]
        archived = grouped[("nsga2", rep["seed"])][rep["point"]]
        assert list(map(int, archived["genes"].split())) == rep["genes"]
        csv_rows = [r for r in timetable_csv if r["solution"] == rep["key"]]
        assert len(csv_rows) == 24
        for csv_row, item in zip(csv_rows, rep["timetable"]):
            assert all(csv_row[k] == str(v) for k, v in item.items())
    for key, name in [
        ("config_sha256", "config.json"),
        ("instance_sha256", "instance.json"),
    ]:
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == manifest[key]
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for document in [ROOT / "README.md", output / "report.md"]:
        text = document.read_text()
        assert not any(ord(c) < 32 and c not in "\n\t" for c in text)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            assert (document.parent / target).exists(), (document, target)
    if repeat:
        from scheduling import Problem, nsga2, weighted_portfolio

        problem = Problem.from_dict(data)
        for r in runs:
            method, seed = r["method"], int(r["seed"])
            result = (
                nsga2(problem, config["common"], seed)
                if method == "nsga2"
                else weighted_portfolio(
                    problem, config["common"], seed, config["weights"]
                )
            )
            expected = grouped[(method, seed)]
            np.testing.assert_array_equal(result["costs"], vectors((method, seed)))
            np.testing.assert_array_equal(
                result["genes"], [list(map(int, v["genes"].split())) for v in expected]
            )
            saved = histories[(method, seed)]
            assert len(saved) == len(result["history"])
            for before, after in zip(saved, result["history"]):
                assert all(float(before[k]) == v for k, v in after.items())
        print(
            f"Exact repetition passed: {len(runs)} runs, {len(history)} history rows."
        )
    print(
        f"Independent audit passed: {len(fronts)} archived schedules, scalar decoding, HV cells, statistics and calendars."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    audit(args.output, args.repeat)
