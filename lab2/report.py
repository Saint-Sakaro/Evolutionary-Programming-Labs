"""Формирование отчёта по маршрутам и статистике запусков."""

import os
from pathlib import Path

import numpy as np
from plotting import method_label


def mst_bound(coordinates):
    """Prim's minimum spanning tree length, a lower bound for any closed tour."""
    coordinates = np.asarray(coordinates)
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    visited = np.zeros(len(distance), dtype=bool)
    closest = np.full(len(distance), np.inf)
    closest[0] = 0
    total = 0.0
    for _ in range(len(distance)):
        node = int(np.argmin(np.where(visited, np.inf, closest)))
        total += closest[node]
        visited[node] = True
        closest = np.minimum(closest, distance[node])
    return total


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def route_text(route):
    if isinstance(route, str):
        route = list(map(int, route.split()))
    return " → ".join(map(str, [0] + list(route) + [0]))


def make_report(output, config, data, rows, summary, examples, best, schedule):
    common = config["common"]
    n = len(data["coordinates_km"]) - 1
    limit = data["time_limit_min"]
    ga_summary = summary[:-1]
    best_ga = min(
        ga_summary,
        key=lambda s: (
            -s["success_rate"],
            s["median"] if s["median"] is not None else np.inf,
            s["mean"] if s["mean"] is not None else np.inf,
        ),
    )
    baseline = summary[-1]
    rescued = sum(
        r["method"].startswith("ga_")
        and r["feasible"]
        and r["elite_violation_min"] > 1e-9
        for r in rows
    )
    archive_conclusion = (
        f"В {rescued} запусках ГА лучший по F маршрут нарушал лимит времени, "
        "но допустимое решение сохранилось в отдельном архиве."
        if rescued
        else "Случаев с допустимым архивом и недопустимым лучшим маршрутом по F в серии не было."
    )
    table = "\n".join(
        f"| {method_label(s['method'])} | {s['feasible_runs']}/{s['runs']} | {fmt(s['best'])} | "
        f"{fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['std'])} | {fmt(s['worst'])} |"
        for s in summary
    )
    time_table = "\n".join(
        f"| {method_label(s['method'])} | {s['evaluated_feasible_fraction']:.2%} | {s['mean_seconds']:.4f} |"
        for s in summary
    )
    pair_rows = []
    p0, p1 = config["penalties"]

    def paired(left, right, label):
        a = {r["seed"]: r for r in rows if r["method"] == left}
        b = {r["seed"]: r for r in rows if r["method"] == right}
        seeds = [s for s in config["seeds"] if a[s]["feasible"] and b[s]["feasible"]]
        deltas = [b[s]["distance_km"] - a[s]["distance_km"] for s in seeds]
        pair_rows.append(
            f"| {label} | {len(seeds)} | {fmt(float(np.mean(deltas)) if deltas else None)} |"
        )

    for mutation in config["mutations"]:
        paired(
            f"ga_{mutation}_p{p0:g}",
            f"ga_{mutation}_p{p1:g}",
            f"λ: {p0:g} → {p1:g}, {'обмен' if mutation == 'swap' else 'инверсия'}",
        )
    for penalty in config["penalties"]:
        paired(
            f"ga_swap_p{penalty:g}",
            f"ga_inversion_p{penalty:g}",
            f"Обмен → инверсия, λ={penalty:g}",
        )
    penalty_effects = []
    for mutation in config["mutations"]:
        left = next(s for s in summary if s["method"] == f"ga_{mutation}_p{p0:g}")
        right = next(s for s in summary if s["method"] == f"ga_{mutation}_p{p1:g}")
        label = "обмене" if mutation == "swap" else "инверсии"
        if left["feasible_runs"] == right["feasible_runs"]:
            effect = (
                f"При {label} оба штрафа дали {left['feasible_runs']} допустимых "
                f"запусков из {left['runs']}."
            )
        else:
            effect = (
                f"При {label} увеличение λ с {p0:g} до {p1:g} снизило число "
                f"допустимых запусков с {left['feasible_runs']} до {right['feasible_runs']}."
                if right["feasible_runs"] < left["feasible_runs"]
                else f"При {label} увеличение λ с {p0:g} до {p1:g} повысило число "
                f"допустимых запусков с {left['feasible_runs']} до {right['feasible_runs']}."
            )
        if left["infeasible_elite_runs"] != right["infeasible_elite_runs"]:
            effect += (
                f" Число запусков с недопустимым лучшим маршрутом по F "
                f"изменилось с {left['infeasible_elite_runs']} до {right['infeasible_elite_runs']}."
            )
        effect += f" Медианы длин: {fmt(left['median'])} и {fmt(right['median'])} км."
        penalty_effects.append(effect)
    example_table = "\n".join(
        f"| {e['name']} | {e['distance_km']:.3f} | {e['time_min']:.3f} | "
        f"{e['violation_min']:.3f} | {'Да' if e['feasible'] else 'Нет'} |"
        for e in examples
    )
    example_details = "\n\n".join(
        f"**{e['name']}:** {route_text(e['route'])}." for e in examples
    )
    schedule_table = "\n".join(
        f"| {s['visit']} | {s['node']} | {s['leg_km']:.3f} | {s['travel_min']:.3f} | "
        f"{s['arrival_min']:.3f} | {s['service_min']:g} | {s['departure_min']:.3f} |"
        for s in schedule
    )
    mst = mst_bound(data["coordinates_km"])
    baseline_analysis = (
        f"Медиана выбранной конфигурации ГА — {best_ga['median']:.3f} км, "
        f"эвристики — {baseline['median']:.3f} км. "
        + (
            "В этой серии ГА дал меньшую медианную длину."
            if best_ga["median"] < baseline["median"]
            else "В этой серии ГА не улучшил медианную длину относительно эвристики."
        )
        if best_ga["median"] is not None and baseline["median"] is not None
        else "Часть методов не нашла допустимые решения; это отражено в доле успешных запусков."
    )
    readme = Path(
        os.path.relpath(Path(__file__).resolve().parent / "README.md", output.resolve())
    ).as_posix()
    full_generations, remainder = divmod(
        common["budget"] - common["population_size"], common["population_size"]
    )
    budget_text = (
        f"{common['population_size']} начальных оценок + {full_generations} поколений "
        f"по {common['population_size']} оценок"
    )
    if remainder:
        budget_text += f" + {remainder} оценок неполного поколения"
    text = rf"""# Лабораторная № 2 — маршрут доставки

Белов Фёдор, группа Б24-527, номер 3, вариант 12, 2026 год.
V=1+((17·3+7·2+26) mod 20)=12.

## Задача

Один автомобиль выезжает со склада 0, посещает {n} адресов по одному разу
и возвращается на склад. Нужно минимизировать расстояние при ограничении
полной продолжительности рейса.

Генотип — перестановка клиентов $\pi=(\pi_1,\ldots,\pi_n)$.
Фенотип — маршрут со складом в начале и конце: $\pi_0=\pi_{{n+1}}=0$.

$$L(\pi)=\sum_{{k=0}}^n d_{{\pi_k,\pi_{{k+1}}}}\to\min,$$

$$T(\pi)=\sum_{{k=0}}^n \tau_{{\pi_k,\pi_{{k+1}}}}+\sum_{{i=1}}^n s_i
\le {limit:g}\ \text{{мин}}.$$

Здесь dᵢⱼ — евклидово расстояние, τᵢⱼ — направленное время проезда,
sᵢ — обслуживание. Вместимость автомобиля и временные окна не рассматриваются.

## Данные

Набор сгенерирован с seed {data["generation_seed"]} и сохранён в
[instance.json](instance.json).

- Склад — (20,20) км, клиенты равномерно распределены на [0,40]² км.
- Номинальная скорость — 0,6 км/мин.
- Время дуги: $\tau_{{ij}}=d_{{ij}}c_{{ij}}/0.6$, где cᵢⱼ независимо выбирается
  из U(0,8;1,8). Матрица несимметрична, её диагональ нулевая.
- Обслуживание — целое число от 3 до 8 минут, на складе — 0.

Для выбора лимита построен маршрут ближайшего соседа по времени и улучшен
инверсиями сегментов. Его время умножено на 1,10 и округлено вверх.
Так получен хотя бы один допустимый маршрут. В стартовую популяцию и результаты
алгоритмов этот маршрут не добавляется.

Коэффициенты нагрузки фиксированы на весь рейс.
Время проезда не является постоянным множителем длины.

## Алгоритм и ограничения

Структурную допустимость обеспечивает перестановка клиентов.
Временное ограничение учитывается штрафом:

$$F_\lambda(\pi)=L(\pi)+\lambda\max(0,T(\pi)-T_{{\max}}).$$

Сравниваются λ={p0:g} и λ={p1:g} км/мин.
Конечный штраф может оставлять недопустимую особь лучшей по F.
Поэтому отдельно сохраняются штрафная элита и лучший допустимый маршрут по L.
Если допустимый маршрут не найден, запуск помечается неуспешным.

1. Создать случайные перестановки и оценить L, T и F.
2. Турнирами выбрать родителей.
3. Применить OX и выбранную мутацию.
4. Оценить потомков, обновить элиту и допустимый архив.
5. Заменить худшего потомка штрафной элитой.
6. Повторять до исчерпания бюджета.

![Блок-схема](flowchart.png)

**OX** копирует сегмент первого родителя. Остальные позиции заполняются
в циклическом порядке второго родителя, без уже скопированных клиентов.
Так сохраняются порядок и множество адресов.

**Обмен** меняет местами два клиента. **Инверсия** разворачивает сегмент.
Оба оператора сохраняют перестановку. При направленной матрице инверсия меняет
внутренние дуги, поэтому полное время пересчитывается заново.

Повтор клиента, пропуск адреса, склад внутри хромосомы или неизвестный номер
отклоняются оценщиком. Допуск при проверке времени — 10⁻⁹ минуты.

### Четыре примера маршрутов

Все примеры — корректные перестановки. В последних двух превышен лимит времени.

| Пример | Длина, км | Время, мин | Превышение, мин | Допустим |
|---|---:|---:|---:|---|
{example_table}

{example_details}

## Эксперимент

| Параметр | Значение |
|---|---|
| Популяция | {common["population_size"]} |
| Размер турнира | {common["tournament_size"]} |
| Вероятность OX | {common["crossover_probability"]} на пару |
| Вероятность мутации | {common["mutation_probability"]} на потомка |
| Мутации | Обмен, инверсия |
| Штрафы | {p0:g} и {p1:g} км/мин |
| Бюджет | {common["budget"]} оценок на запуск |
| Seed | {", ".join(map(str, config["seeds"]))} |
| Базовый метод | Рандомизированный ближайший сосед, k={config["baseline_candidate_list"]} |

План 2×2: при сравнении штрафов фиксируется мутация и наоборот.
В каждой конфигурации {len(config["seeds"])} запусков.
Начальные популяции ГА при одном seed совпадают.

Бюджет ГА: {budget_text}. В него входят инициализация и недопустимые потомки.
Сохранённые оценки элиты повторно не вычисляются.

Базовый метод на каждом шаге случайно выбирает одного из k ближайших
по времени непосещённых клиентов. Полный маршрут проверяется на допустимость.
Все {common["budget"]} попыток, включая неудачные, входят в бюджет.
Стоимость построения кандидатов различается, поэтому дополнительно измеряется время.

## Результаты

Статистика длины рассчитана только по допустимым итогам.
Число успешных запусков указано отдельно. Стандартное отклонение выборочное.

| Метод | Допустимые запуски | Лучшее, км | Среднее, км | Медиана, км | Станд. откл., км | Худшее, км |
|---|---:|---:|---:|---:|---:|---:|
{table}

| Метод | Доля допустимых среди оценённых | Среднее время, с |
|---|---:|---:|
{time_table}

Время включает поиск и работу операторов, без записи файлов и графиков.

![Сходимость и доля успешных запусков](convergence.png)

Средняя длина слева рассчитана по запускам, уже нашедшим допустимый маршрут;
их состав на ранних шагах меняется. Заливка — min–max.
Справа показана доля таких запусков.

![Итоги серии](comparison.png)

Каждая точка — допустимый итог запуска, чёрный отрезок — медиана.

### Влияние штрафа и мутации

| Изменение | Пар с двумя допустимыми итогами | Средняя разность после − до, км |
|---|---:|---:|
{chr(10).join(pair_rows)}

Отрицательная разность означает сокращение маршрута.
В пары входят только два допустимых результата; доли успеха сравниваются отдельно.

{(chr(10) * 2).join(penalty_effects)}

Число недопустимых штрафных элит и число успешных запусков — разные показатели.

По доле успехов, затем по медиане и среднему выбран вариант
{method_label(best_ga["method"])}. {baseline_analysis}

## Лучший маршрут

Метод: {method_label(best["method"])}, seed={best["seed"]}.

{route_text(best["route"])}

Длина — {best["distance_km"]:.6f} км.
Время — {best["time_min"]:.6f} мин, запас до лимита — {limit - best["time_min"]:.6f} мин.

![Карта маршрута](best_route.png)

| Порядок | Адрес | Участок, км | Проезд, мин | Прибытие, мин | Обслуживание, мин | Отправление, мин |
|---|---:|---:|---:|---:|---:|---:|
{schedule_table}

Последняя строка — возврат на склад. Полная точность — в [best_route.csv](best_route.csv).

Минимальное остовное дерево имеет длину {mst:.6f} км и задаёт нижнюю границу:
если удалить ребро тура, останется остовное дерево.
Найденный маршрут длиннее этой границы на {100 * (best["distance_km"] / mst - 1):.2f}%.
Это не установленное отклонение от неизвестного оптимума.

## Выводы

Конфигурация «{method_label(best_ga["method"])}»: доля успешных запусков —
{best_ga["success_rate"]:.0%}, медианная длина — {fmt(best_ga["median"])} км.
{archive_conclusion}

Сравнение проведено на одном синтетическом наборе, при одном бюджете и двух
штрафах. Статистическая значимость различий не проверялась.
Точный оптимум не вычислялся.

## Материалы

[Запуск]({readme}), [параметры](config.json), [данные](instance.json),
[примеры](examples.json), [запуски](runs.csv), [статистика](summary.csv),
[истории](history.csv), [версии и хеши](manifest.json).

Задание: Р. В. Душкин, «Эволюционное программирование и генетические алгоритмы»,
лабораторная № 2, вариант 12.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
