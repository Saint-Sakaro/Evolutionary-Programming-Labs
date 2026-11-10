"""Графики поиска и найденных маршрутов."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def method_label(name):
    if name == "randomized_greedy":
        return "Рандомизированный жадный поиск"
    _, mutation, penalty = name.split("_")
    return f"{'Обмен' if mutation == 'swap' else 'Инверсия'}, λ={penalty[1:]}"


def save(fig, output, name):
    fig.savefig(output / f"{name}.png")
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def make_plots(output, config, problem, rows, histories, best):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    names = list(dict.fromkeys(r["method"] for r in rows))
    colors = ["#1764ab", "#7b51a0", "#d1761c", "#1f886c", "#697780"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name, color in zip(names, colors):
        by_seed = [
            [h for h in histories if h["method"] == name and h["seed"] == seed]
            for seed in config["seeds"]
        ]
        values = np.array(
            [
                [
                    np.nan
                    if h["best_feasible_distance"] is None
                    else h["best_feasible_distance"]
                    for h in series
                ]
                for series in by_seed
            ]
        )
        has = np.isfinite(values)
        count = has.sum(axis=0)
        mean = np.divide(
            np.nansum(values, axis=0),
            count,
            out=np.full(values.shape[1], np.nan),
            where=count > 0,
        )
        low = np.min(np.where(has, values, np.inf), axis=0)
        high = np.max(np.where(has, values, -np.inf), axis=0)
        low[count == 0], high[count == 0] = np.nan, np.nan
        x = [h["evaluations"] for h in by_seed[0]]
        axes[0].plot(x, mean, color=color, label=method_label(name))
        axes[0].fill_between(x, low, high, color=color, alpha=0.08)
        axes[1].plot(
            x, count / len(config["seeds"]), color=color, label=method_label(name)
        )
    axes[0].set(
        xlabel="Число оценённых маршрутов",
        ylabel="Длина, км",
        title="Лучший допустимый маршрут: среднее и min–max",
    )
    axes[1].set(
        xlabel="Число оценённых маршрутов",
        ylabel="Доля запусков",
        title="Найден хотя бы один допустимый маршрут",
        ylim=(-0.03, 1.03),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        "По 20 запусков; средняя длина условна по запускам с допустимым решением"
    )
    save(fig, output, "convergence")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for i, (name, color) in enumerate(zip(names, colors)):
        current = [r for r in rows if r["method"] == name]
        valid = [r["distance_km"] for r in current if r["feasible"]]
        if valid:
            axes[0].scatter(
                i + np.linspace(-0.13, 0.13, len(valid)), valid, color=color, s=24
            )
            axes[0].plot(
                [i - 0.23, i + 0.23], [np.median(valid)] * 2, color="black", linewidth=2
            )
        axes[1].bar(
            i, np.mean([r["feasible"] for r in current]), color=color, alpha=0.8
        )
    labels = [
        "Жадный\nпоиск"
        if name == "randomized_greedy"
        else method_label(name).replace(", ", "\n")
        for name in names
    ]
    for axis in axes:
        axis.set_xticks(range(len(names)), labels, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set(
        ylabel="Длина, км", title="Итоговые допустимые маршруты; чёрная линия — медиана"
    )
    axes[1].set(
        ylabel="Доля запусков",
        title="Успешное нахождение допустимого решения",
        ylim=(0, 1.1),
    )
    save(fig, output, "comparison")

    fig, axis = plt.subplots(figsize=(8, 7.5), layout="constrained")
    full = [0] + list(map(int, best["route"].split())) + [0]
    coordinates = problem.coordinates
    for a, b in zip(full, full[1:]):
        axis.annotate(
            "",
            xy=coordinates[b],
            xytext=coordinates[a],
            arrowprops=dict(arrowstyle="->", color="#1764ab", lw=1.2, alpha=0.75),
        )
    axis.scatter(
        coordinates[1:, 0], coordinates[1:, 1], color="#1764ab", s=45, zorder=3
    )
    axis.scatter(
        *coordinates[0], marker="s", s=120, color="#cc651c", label="Склад 0", zorder=4
    )
    for i, (x, y) in enumerate(coordinates):
        axis.annotate(
            str(i), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9
        )
    axis.set_aspect("equal")
    axis.set(
        xlabel="x, км",
        ylabel="y, км",
        title=f"Лучший маршрут: {best['distance_km']:.2f} км; "
        f"{best['time_min']:.2f} ≤ {problem.limit:g} мин",
    )
    axis.margins(0.12)
    axis.grid(alpha=0.2)
    axis.legend()
    save(fig, output, "best_route")

    from matplotlib.patches import FancyBboxPatch

    fig, axis = plt.subplots(figsize=(8, 9), layout="constrained")
    axis.set(xlim=(0, 10), ylim=(-0.8, 11.6))
    axis.axis("off")
    boxes = [
        (10.5, "Данные, параметры, seed → случайные перестановки"),
        (9.0, "Оценить длину, время и штраф\nОбновить элиту и допустимый архив"),
        (7.5, "Бюджет исчерпан?"),
        (6.0, "Турнир → OX → обмен или инверсия"),
        (4.5, "Оценить потомков, увеличить счётчик"),
        (3.0, "Обновить скалярную элиту и допустимый архив"),
        (1.5, "Элитизм → новое поколение"),
    ]
    for y, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (0.8, y - 0.42),
                6.8,
                0.84,
                boxstyle="round,pad=0.12",
                facecolor="#eef4fa",
                edgecolor="#1764ab",
            )
        )
        axis.text(4.2, y, label, ha="center", va="center", fontsize=9)
    for (y, _), (next_y, _) in zip(boxes, boxes[1:]):
        axis.annotate(
            "",
            xy=(4.2, next_y + 0.54),
            xytext=(4.2, y - 0.54),
            arrowprops=dict(arrowstyle="->", color="#1764ab"),
        )
    axis.text(4.45, 6.75, "Нет", fontsize=9)
    axis.plot([7.75, 8.9, 8.9], [7.5, 7.5, -0.1], color="#1764ab")
    axis.annotate(
        "",
        xy=(7.6, -0.1),
        xytext=(8.9, -0.1),
        arrowprops=dict(arrowstyle="->", color="#1764ab"),
    )
    axis.text(8.0, 7.7, "Да", fontsize=9)
    axis.add_patch(
        FancyBboxPatch(
            (0.8, -0.52),
            6.8,
            0.84,
            boxstyle="round,pad=0.12",
            facecolor="#eaf3e5",
            edgecolor="#699143",
        )
    )
    axis.text(
        4.2,
        -0.1,
        "Допустимый маршрут или явный признак неуспеха",
        ha="center",
        va="center",
        fontsize=9,
    )
    axis.plot([0.25, 0.25, 4.2], [1.5, 8.25, 8.25], color="#1764ab")
    axis.plot([0.25, 0.68], [1.5, 1.5], color="#1764ab")
    axis.annotate(
        "",
        xy=(4.2, 8.04),
        xytext=(4.2, 8.25),
        arrowprops=dict(arrowstyle="->", color="#1764ab"),
    )
    save(fig, output, "flowchart")
