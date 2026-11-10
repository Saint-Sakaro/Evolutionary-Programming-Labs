"""Формирование отчёта по сохранённым результатам."""

import os
from math import sqrt
from pathlib import Path


def make_report(output, config, rows, summary):
    problem, common = config["problem"], config["common"]
    configurations = config["configurations"]
    labels = {c["name"]: f"ГА, σ={c['mutation_sigma']:g}" for c in configurations}
    labels["random_search"] = "Случайный поиск"
    best = min(rows, key=lambda r: r["best_f"])
    ranked = sorted(summary[:2], key=lambda s: s["median"])
    better, worse = ranked
    random = summary[-1]
    full_generations, remainder = divmod(
        common["budget"] - common["population_size"], common["population_size"]
    )
    budget_description = (
        f"{common['population_size']} начальных оценок и {full_generations} полных "
        f"поколений по {common['population_size']} оценок"
    )
    if remainder:
        budget_description += f", затем неполное поколение из {remainder} оценок"
    readme_target = Path(
        os.path.relpath(Path(__file__).resolve().parent / "README.md", output.resolve())
    ).as_posix()
    better_values = {
        r["seed"]: r["best_f"] for r in rows if r["method"] == better["method"]
    }
    worse_values = {
        r["seed"]: r["best_f"] for r in rows if r["method"] == worse["method"]
    }
    paired_wins = sum(better_values[s] < worse_values[s] for s in config["seeds"])
    stats_table = "\n".join(
        f"| {labels[s['method']]} | {s['best']:.6e} | {s['mean']:.6e} | {s['median']:.6e} | "
        f"{s['std']:.6e} | {s['worst']:.6e} | {s['success_rate']:.0%} |"
        for s in summary
    )
    time_table = "\n".join(
        f"| {labels[s['method']]} | {s['evaluations']} | {s['mean_seconds']:.4f} |"
        for s in summary
    )
    coordinates = "\n".join(
        f"| {i} | {best[f'x{i}']:.12f} | {sqrt(i):.12f} | "
        f"{abs(abs(best[f'x{i}']) - sqrt(i)):.6e} |"
        for i in range(1, problem["dimension"] + 1)
    )
    text = rf"""# Лабораторная № 1 — функция Цин

Белов Фёдор, группа Б24-527, номер 3, вариант 12, 2026 год.

## Задача

$$V=1+((17\cdot3+7\cdot2+26)\bmod20)=12,\qquad d=5+(V\bmod6)=5.$$

Найти минимум функции Цин:

$$f(x)=\sum_{{i=1}}^5(x_i^2-i)^2,\qquad x_i\in[-500,500].$$

Каждое слагаемое неотрицательно. При $x_i=\pm\sqrt{{i}}$ все они равны нулю,
поэтому $f^*=0$. Таких точек $2^5=32$.

Особь — вектор из пяти вещественных координат; генотип и фенотип совпадают.
Для отбора сравниваются значения f(x): меньшее лучше.
Начальные точки выбираются равномерно на всей области. Корни используются
только для проверки результата.

## Алгоритм

1. Создать популяцию и оценить функцию.
2. Выбрать родителей независимыми турнирами с возвращением.
3. Выполнить BLX-α и гауссовскую мутацию.
4. Отразить координаты, вышедшие за границы.
5. Оценить потомков, обновить лучшее решение и заменить им худшего потомка.
6. Повторять до исчерпания бюджета.

![Блок-схема](flowchart.png)

Для координат родителей aᵢ и bᵢ, где $r_i=|a_i-b_i|$, BLX-α задаёт:

$$c_i\sim U\left(\min(a_i,b_i)-\alpha r_i,\ \max(a_i,b_i)+\alpha r_i\right).$$

Оператор сохраняет вещественное представление и расширяет интервал поиска
вокруг родителей. Покомпонентная рекомбинация подходит для разделимой функции Цин.
BLX-α также выполняет дополнительное задание.

При мутации выбранной координаты прибавляется $\varepsilon_i\sim N(0,\sigma^2)$.
Для отражения, если l и u — границы, w=u−l:

$$t=(x-l)\bmod(2w),\qquad x'=l+\min(t,2w-t).$$

Формула обрабатывает и многократный выход за границы.
Значение сохраняемой элиты берётся из кеша.

## Эксперимент

| Параметр | Значение |
|---|---|
| Популяция | {common["population_size"]} |
| Размер турнира | {common["tournament_size"]} |
| Вероятность кроссовера | {common["crossover_probability"]} на пару |
| BLX-α | α={common["blx_alpha"]} |
| Вероятность мутации | {common["mutation_probability"]} на координату |
| Масштабы мутации | {configurations[0]["mutation_sigma"]:g} и {configurations[1]["mutation_sigma"]:g} |
| Элитизм | Одна особь |
| Бюджет | {common["budget"]} оценок, включая инициализацию |
| Seed | {", ".join(map(str, config["seeds"]))} |

Между двумя конфигурациями меняется только σ. Базовый метод проверяет
{common["budget"]} независимых равномерных точек и сохраняет лучшую.
При одном seed у ГА и случайного поиска совпадает начальная выборка.

Бюджет ГА: {budget_description}; всего {common["budget"]} оценок.
Поколение 0 — инициализация. Проведено {len(rows)} запусков.

Успехом считается f(x)≤{config["success_threshold"]:g}.
Этот порог выбран для эксперимента и не используется для досрочной остановки.

## Результаты

В таблице — статистика лучших итоговых значений по {len(config["seeds"])} seed.
Стандартное отклонение выборочное, с делителем n−1.

| Метод | Лучшее | Среднее | Медиана | Станд. откл. | Худшее | Успех |
|---|---:|---:|---:|---:|---:|---:|
{stats_table}

| Метод | Оценок | Среднее время, с |
|---|---:|---:|
{time_table}

Время включает поиск, без сохранения файлов и построения графиков.

![Сходимость](convergence.png)

Линии показывают среднее лучшее значение; заливка — min–max по seed.
Слева — поколения ГА, справа — общий бюджет всех методов.
Вертикальная шкала логарифмическая.

![Итоги отдельных запусков](final_values.png)

Каждая точка — запуск, чёрный отрезок — медиана.

### Лучшая точка

Метод: {labels[best["method"]]}, seed={best["seed"]}.
Значение функции: {best["best_f"]:.12e}.

| i | xᵢ | √i | Ошибка модуля координаты |
|---|---:|---:|---:|
{coordinates}

Сравниваются модули координат, поскольку оба знака корня дают минимум.

## Выводы

Медиана для {labels[better["method"]]} — {better["median"]:.6e},
для {labels[worse["method"]]} — {worse["median"]:.6e}.
В парных сравнениях seed первая конфигурация получила меньшую ошибку
в {paired_wins} из {len(config["seeds"])} случаев.

Медиана случайного поиска — {random["median"]:.6e}.
Ему мешает размер области: все пять координат должны одновременно попасть
в небольшие окрестности корней. ГА сохраняет удачные координаты и уточняет их.

У функции Цин нет плохих локальных минимумов: производная слагаемого равна
$4x_i(x_i^2-i)$; при нуле у него максимум, при ±√i — минимумы.
Поэтому результат сравнения со случайным поиском не доказывает преимущество
над локальными методами.

Доля успехов выбранной конфигурации — {better["success_rate"]:.0%};
худшая ошибка — {better["worst"]:.6e}.
Вывод относится к одной функции, двум масштабам и заданному бюджету.

## Материалы

[Инструкция запуска]({readme_target}).
Параметры и версии: [config.json](config.json), [manifest.json](manifest.json).
Данные: [runs.csv](runs.csv), [summary.csv](summary.csv), [history.csv](history.csv).

Задание: Р. В. Душкин, «Эволюционное программирование и генетические алгоритмы»,
лабораторная № 1; формула Цин — пункт 12 сопутствующего файла.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
