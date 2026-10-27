"""Standalone publication/export plots from the actual run records."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_plots(output, config, rows, histories):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 180})
    names = [c['name'] for c in config['configurations']] + ['random_search']
    labels = [f"ГА, σ = {c['mutation_sigma']:g}" for c in config['configurations']] + ['Случайный поиск']
    colors = ['#1764ab', '#cc651c', '#697780']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for index, name in enumerate(names):
        series = [[h for h in histories if h['method'] == name and h['seed'] == seed]
                  for seed in config['seeds']]
        values = np.array([[h['best_f'] for h in run] for run in series])
        # Logarithms only change display, never the saved measurements.
        for axis_index, axis in enumerate(axes):
            if axis_index == 0 and name == 'random_search':
                continue
            x = [h['generation' if axis_index == 0 else 'evaluations'] for h in series[0]]
            axis.plot(x, np.maximum(values.mean(axis=0), 1e-30),
                      color=colors[index], label=labels[index])
            axis.fill_between(x, np.maximum(values.min(axis=0), 1e-30),
                              np.maximum(values.max(axis=0), 1e-30), color=colors[index], alpha=.12)
    for axis in axes:
        axis.set_yscale('log')
        axis.set_ylabel('Лучшее найденное f(x), логарифмическая шкала')
        axis.grid(alpha=.2, which='major')
        axis.legend()
    axes[0].set(xlabel='Поколение (0 — инициализация)', title='Сходимость ГА по поколениям')
    axes[1].set(xlabel='Число вычислений функции', title='Сравнение при одинаковом бюджете')
    fig.suptitle(f"Функция Цин, d = {config['problem']['dimension']}: среднее и диапазон min–max, "
                 f"{len(config['seeds'])} запусков")
    fig.savefig(output / 'convergence.png')
    fig.savefig(output / 'convergence.svg')
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8), layout='constrained')
    jitter = np.linspace(-.12, .12, len(config['seeds']))
    for i, name in enumerate(names):
        values = np.array([r['best_f'] for r in rows if r['method'] == name])
        axis.scatter(i + jitter, np.maximum(values, 1e-30), color=colors[i], alpha=.8, s=24)
        axis.plot([i-.22, i+.22], [np.median(values)] * 2, color='black', linewidth=2,
                  label='Медиана' if i == 0 else None)
    axis.axhline(config['success_threshold'], linestyle='--', color='#699143',
                 label=f"Порог успеха {config['success_threshold']:g}")
    axis.set_xticks(range(len(names)), labels)
    axis.set_yscale('log')
    axis.set(ylabel='Итоговое f(x), логарифмическая шкала', title='Каждая точка — отдельный независимый запуск')
    axis.grid(axis='y', alpha=.2)
    axis.legend()
    fig.savefig(output / 'final_values.png')
    fig.savefig(output / 'final_values.svg')
    plt.close(fig)

    from matplotlib.patches import FancyBboxPatch
    fig, axis = plt.subplots(figsize=(8, 9), layout='constrained')
    axis.set(xlim=(0, 10), ylim=(-.8, 11.6))
    axis.axis('off')
    boxes = [
        (10.5, 'Параметры и seed → начальная популяция'),
        (9.0, 'Оценить популяцию, сохранить лучшего\nСчётчик оценок = размер популяции'),
        (7.5, 'Бюджет исчерпан?'),
        (6.0, 'Турнирный отбор → BLX-α'),
        (4.5, 'Гауссовская мутация → отражение границ'),
        (3.0, 'Оценить потомков, увеличить счётчик\nОбновить лучшее решение'),
        (1.5, 'Элитизм → новая популяция'),
    ]
    for y, label in boxes:
        axis.add_patch(FancyBboxPatch((.8, y-.42), 6.8, .84,
                       boxstyle='round,pad=0.12', facecolor='#eef4fa', edgecolor='#1764ab'))
        axis.text(4.2, y, label, ha='center', va='center', fontsize=10)
    for (y, _), (next_y, _) in zip(boxes, boxes[1:]):
        axis.annotate('', xy=(4.2, next_y+.54), xytext=(4.2, y-.54),
                      arrowprops={'arrowstyle': '->', 'color': '#1764ab'})
    axis.text(4.45, 6.75, 'Нет', fontsize=9)
    axis.annotate('', xy=(7.75, 7.5), xytext=(8.9, 7.5),
                  arrowprops={'arrowstyle': '<-', 'color': '#1764ab'})
    axis.plot([8.9, 8.9], [7.5, -.1], color='#1764ab')
    axis.annotate('', xy=(7.6, -.1), xytext=(8.9, -.1),
                  arrowprops={'arrowstyle': '->', 'color': '#1764ab'})
    axis.text(8.1, 7.7, 'Да', fontsize=9)
    axis.add_patch(FancyBboxPatch((.8, -.52), 6.8, .84,
                   boxstyle='round,pad=0.12', facecolor='#eaf3e5', edgecolor='#699143'))
    axis.text(4.2, -.1, 'Вернуть лучшую точку и историю', ha='center', va='center')
    axis.plot([.25, .25, 4.2], [1.5, 8.25, 8.25], color='#1764ab')
    axis.plot([.25, .68], [1.5, 1.5], color='#1764ab')
    axis.annotate('', xy=(4.2, 8.04), xytext=(4.2, 8.25),
                  arrowprops={'arrowstyle': '->', 'color': '#1764ab'})
    fig.savefig(output / 'flowchart.png')
    fig.savefig(output / 'flowchart.svg')
    plt.close(fig)
