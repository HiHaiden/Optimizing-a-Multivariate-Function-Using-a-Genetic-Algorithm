from deap import base
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
import numpy as np
import time

import algel

# константы задачи
BOUND_LOW, BOUND_UP = -2.048, 2.048
DIMENSIONS = 2  # длина хромосомы, подлежащей оптимизации

# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.4  # вероятность мутации индивидуума
MAX_GENERATIONS = 100  # максимальное количество поколений
HALL_OF_FAME_SIZE = 10  # Зал славы
CROWDING_FACTOR = 20.0

RANDOM_SEED = 42  # зерно для генератора случайных чисел
random.seed(RANDOM_SEED)

# создание класса для описания значения приспособленности особей
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# создание класса для представления каждого индивидуума
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomPoint(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]


def func(individual):
    x1, x2 = individual
    x3, x4 = individual
    f = (100 * ((x2 - (x1 ** 2)) ** 2) + ((1 - x1) ** 2)) + (100 * ((x3 - (x2 ** 2)) ** 2) + ((1 - x2) ** 2))
    return f,

toolbox = base.Toolbox()
# определение функции для генерации случайных значений
toolbox.register("randomPoint", randomPoint, BOUND_LOW, BOUND_UP)
# определение функции для генерации отдельного индивидуума
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
# определение функции для создания начальной популяции
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

# вычисление приспособленности каждой особи на основе func
toolbox.register("evaluate", func)
# отбор особей (турнирный с размером 3)
toolbox.register("select", tools.selTournament, tournsize=3)
# скрещивание особей
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
# мутация
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR,
                 indpb=1.0 / DIMENSIONS)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)


def show(ax, xgrid, ygrid, f):
    ax.clear()
    ax.contour(xgrid, ygrid, f)
    ax.scatter(0, 0, marker='X', color='red', zorder=1)
    ax.scatter(*zip(*population), color='green', s=2, zorder=0)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.2)


x = np.arange(-2.048, 2.048, 3.0)
y = np.arange(-2.048, 2.048, 3.0)
xgrid, ygrid = np.meshgrid(x, y)

f = (100 * ((xgrid - (xgrid ** 2)) ** 2) + ((1 - xgrid) ** 2)) + (100 * ((ygrid - (ygrid ** 2)) ** 2) + ((1 - ygrid) ** 2))

plt.ion()
fig, ax = plt.subplots()
fig.set_size_inches(5, 5)

ax.set_xlim(BOUND_LOW - 1, BOUND_UP + 1)
ax.set_ylim(BOUND_LOW - 1, BOUND_UP + 1)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population, logbook = algel.eaSimpleElitism(population, toolbox,
                                                 cxpb=P_CROSSOVER,
                                                 mutpb=P_MUTATION,
                                                 ngen=MAX_GENERATIONS,
                                                 halloffame=hof,

                                                 stats=stats,
                                                 callback=(show, (ax, xgrid, ygrid, f)),
                                                 verbose=True)

# извлечение статистик:
minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

# вывод лучших результатов:
print("-- Индивидуумы в зале славы = ")
for i in range(HALL_OF_FAME_SIZE):
    print(hof.items[i], sep="\n")

print("-- Лучший индивидуум = ", hof.items[0])
print("-- Лучшая приспособленность = ", hof.items[0].fitness.values[0])

# график статистик:
plt.ioff()
plt.show()

plt.plot(minFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()