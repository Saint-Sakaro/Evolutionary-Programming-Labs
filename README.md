# Эволюционное программирование и генетические алгоритмы

Белов Фёдор, группа Б24-527, номер в списке 3, вариант 12, 2026 год.

V = 1 + ((17 × 3 + 7 × 2 + 26) mod 20) = 12.

| № | Работа | Запуск | Отчёт | Проверки |
|---|---|---|---|---|
| 1 | Генетический алгоритм для функции Цин | [README](lab1/README.md) | [Отчёт](lab1/results/report.md) | [Тесты](lab1/CHECK.md) |
| 2 | Маршрут доставки с ограничением времени | [README](lab2/README.md) | [Отчёт](lab2/results/report.md) | [Тесты](lab2/CHECK.md) |
| 3 | Многокритериальное расписание, NSGA-II | [README](lab3/README.md) | [Отчёт](lab3/results/report.md) | [Тесты](lab3/CHECK.md) |

## Установка

Нужен Python 3.12 или новее. Из корня репозитория:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

В Windows окружение активируется командой .venv\Scripts\activate.
Версии зависимостей закреплены в requirements.txt.
NumPy используется для расчётов и случайных чисел, Matplotlib — для графиков.

## Запуск

~~~bash
python lab1/experiment.py
python lab2/experiment.py
python lab3/experiment.py
~~~

Параметры и seed заданы в config.json каждой работы. Отчёты и CSV уже лежат
в results. Новый запуск перезаписывает эту папку; для отдельной серии добавьте
--output labN/results_repeat, заменив N номером работы.

## Проверка

~~~bash
python -m unittest discover -s lab1/tests -v
python -m unittest discover -s lab2/tests -v
python -m unittest discover -s lab3/tests -v
python lab2/verify_results.py
python lab3/verify_results.py
~~~

Опция --repeat у verify_results.py повторяет всю серию и сравнивает результаты
и истории с сохранёнными CSV. Время выполнения при повторении может отличаться.
