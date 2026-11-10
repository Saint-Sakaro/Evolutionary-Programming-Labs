"""Генерация занятий и пожеланий по времени."""

import argparse
import json
from pathlib import Path

import numpy as np
from scheduling import Problem, evaluate

ROOT = Path(__file__).resolve().parent


def generate(seed=5270313):
    rng = np.random.default_rng(seed)
    topics = [
        "Программирование",
        "Дискретная математика",
        "Алгоритмы",
        "Английский язык",
        "Базы данных",
        "Проектный семинар",
    ]
    events = []
    for group in range(4):
        teachers = [(group + 2 * j) % 4 for j in range(6)]
        if group in (2, 3):
            teachers[2], teachers[3] = teachers[3], teachers[2]
        days = [
            group,
            group,
            (group + 2) % 5,
            (group + 2) % 5,
            (group + 4) % 5,
            (group + 4) % 5,
        ]
        periods = [0, 3, 1, 4, 4, 5]
        # Перемешиваем только названия предметов.
        titles = rng.permutation(topics)
        for j in range(6):
            events.append(
                dict(
                    id=len(events),
                    group=group,
                    teacher=teachers[j],
                    title=str(titles[j]),
                    requested_slot=6 * days[j] + periods[j],
                )
            )
    data = dict(
        generation_seed=seed,
        days=5,
        periods=6,
        rooms=2,
        late_from=4,
        day_names=["Понедельник", "Вторник", "Среда", "Четверг", "Пятница"],
        period_starts=["09:00", "10:40", "12:20", "14:00", "15:40", "17:20"],
        groups=["Г1", "Г2", "Г3", "Г4"],
        teachers=["П1", "П2", "П3", "П4"],
        room_names=["Аудитория 1", "Аудитория 2"],
        events=events,
        description="Structured synthetic requests create a known tradeoff: exact requests have gaps and late classes.",
    )
    problem = Problem.from_dict(data)
    costs, assigned, _ = evaluate(problem, problem.requested[None])
    assert assigned[0].tolist() == problem.requested.tolist()
    assert costs[0].tolist() == [0, 16, 12]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5270313)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "instance.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(generate(args.seed), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output}: 24 events; request schedule objectives (0,16,12)")
