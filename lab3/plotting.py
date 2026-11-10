"""Графики Парето, статистики и расписаний."""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

LABELS = {"nsga2": "NSGA-II", "weighted_sum": "Взвешенная сумма"}


def save(fig, output, name):
    fig.savefig(output / f"{name}.png")
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def make_plots(output, config, data, runs, history, results, representatives, seed):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    colors = {"nsga2": "#1764ab", "weighted_sum": "#cc651c"}
    objectives = ["Конфликты заявок", "Окна (пустые пары)", "Поздние занятия"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for axis, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        for method in ("weighted_sum", "nsga2"):
            f = results[(method, seed)]["costs"]
            if method == "nsga2":
                axis.scatter(
                    f[:, a],
                    f[:, b],
                    facecolors="none",
                    edgecolors=colors[method],
                    s=50,
                    linewidths=1.4,
                    label=LABELS[method],
                )
            else:
                axis.scatter(
                    f[:, a],
                    f[:, b],
                    color=colors[method],
                    marker="x",
                    s=35,
                    alpha=0.7,
                    label=LABELS[method],
                )
        for i, rep in enumerate(representatives, 1):
            value = [rep["conflicts"], rep["windows"], rep["late"]]
            axis.scatter(
                value[a], value[b], color="#a52237", marker="*", s=90, zorder=5
            )
            axis.annotate(
                str(i),
                (value[a], value[b]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="#a52237",
            )
        axis.set(xlabel=objectives[a], ylabel=objectives[b])
        axis.grid(alpha=0.2)
        axis.margins(0.12)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Проекции приближённых фронтов, seed={seed}; номера — выбранные расписания"
    )
    save(fig, output, "pareto")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    grid = np.array(sorted({0} | {h["evaluations"] for h in history}))
    for i, method in enumerate(("nsga2", "weighted_sum")):
        curves = []
        for s in config["seeds"]:
            h = [h for h in history if h["method"] == method and h["seed"] == s]
            times = np.array([0] + [v["evaluations"] for v in h])
            hv = np.array([0.0] + [v["hypervolume"] for v in h])
            curves.append(hv[np.searchsorted(times, grid, side="right") - 1])
        curves = np.array(curves)
        axes[0].plot(
            grid, curves.mean(axis=0), color=colors[method], label=LABELS[method]
        )
        axes[0].fill_between(
            grid,
            curves.min(axis=0),
            curves.max(axis=0),
            color=colors[method],
            alpha=0.12,
        )
        values = [r["hypervolume"] for r in runs if r["method"] == method]
        axes[1].scatter(
            i + np.linspace(-0.12, 0.12, len(values)),
            values,
            color=colors[method],
            s=24,
        )
        axes[1].plot([i - 0.22, i + 0.22], [np.median(values)] * 2, color="black", lw=2)
    axes[0].set(
        xlabel="Число оценок расписания",
        ylabel="Нормированный гиперобъём",
        title="Среднее и диапазон min–max по 20 запускам",
    )
    axes[0].legend()
    axes[1].set_xticks([0, 1], [LABELS["nsga2"], LABELS["weighted_sum"]])
    axes[1].set(
        ylabel="Итоговый нормированный гиперобъём",
        title="Каждая точка — запуск; чёрный отрезок — медиана",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    save(fig, output, "quality")

    for index, rep in enumerate(representatives, 1):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), layout="constrained")
        for group, axis in enumerate(axes.ravel()):
            axis.set(xlim=(0, 5), ylim=(6, 0))
            axis.axhspan(4, 6, color="#fff1df", alpha=0.8)
            for x in range(6):
                axis.axvline(x, color="#d7dce1", lw=0.7)
            for y in range(7):
                axis.axhline(y, color="#d7dce1", lw=0.7)
            for item in rep["timetable"]:
                if item["group"] != data["groups"][group]:
                    continue
                day, period = item["day"], item["period"]
                axis.add_patch(
                    Rectangle(
                        (day + 0.03, period + 0.03),
                        0.94,
                        0.94,
                        facecolor="#e7f0fa",
                        edgecolor="#1764ab",
                        lw=0.8,
                    )
                )
                title = "\n".join(textwrap.wrap(item["title"], 18))
                text = f"{title}\n{item['teacher']}, А{item['room'] + 1}"
                axis.text(
                    day + 0.5,
                    period + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
            axis.set_xticks(np.arange(5) + 0.5, ["Пн", "Вт", "Ср", "Чт", "Пт"])
            axis.set_yticks(np.arange(6) + 0.5, data["period_starts"], fontsize=9)
            axis.set_title(f"Группа {data['groups'][group]}")
        fig.suptitle(
            f"{index}. {rep['label']} — нарушено заявок: {rep['conflicts']}, "
            f"окон-пар: {rep['windows']}, поздних занятий: {rep['late']}\n"
            "Оранжевая область — поздние пары; время начала указано слева",
            fontsize=13,
        )
        save(fig, output, f"schedule_{rep['key']}")

    fig, axis = plt.subplots(figsize=(8, 9), layout="constrained")
    axis.set(xlim=(0, 10), ylim=(-0.8, 11.6))
    axis.axis("off")
    boxes = [
        (10.5, "Начальная популяция → декодер без пересечений"),
        (9.0, "Оценить три критерия → обновить внешний архив"),
        (7.5, "Бюджет исчерпан?"),
        (
            6.0,
            "Турнир по рангу и crowding distance\nКроссовер → мутация → декодирование",
        ),
        (4.5, "Оценить потомков → обновить внешний архив"),
        (
            3.0,
            "Объединить родителей и потомков\nНедоминируемая сортировка, crowding distance",
        ),
        (1.5, "Элитный отбор N особей по фронтам"),
    ]
    for y, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (0.8, y - 0.42),
                6.8,
                0.84,
                boxstyle="round,pad=.12",
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
    axis.text(8.0, 7.7, "Да", fontsize=9)
    axis.annotate(
        "",
        xy=(7.6, -0.1),
        xytext=(8.9, -0.1),
        arrowprops=dict(arrowstyle="->", color="#1764ab"),
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.8, -0.52),
            6.8,
            0.84,
            boxstyle="round,pad=.12",
            facecolor="#eaf3e5",
            edgecolor="#699143",
        )
    )
    axis.text(
        4.2,
        -0.1,
        "Вернуть приближённый фронт и расписания",
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
