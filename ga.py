import random
from deap import base, creator, tools, algorithms
from copy import deepcopy
from model import get_accuracy
import numpy as np
import pandas as pd


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

p_mutation = 0.2
p_cross = 0.8
n_generation = 2
n_run = 2
population_size = 100
tournament_size = 2


init_weight_distribution = ['gaussian', 'uniform', 'constant']
init_weight_scale = [0.5, 1.5]
n_layer = [1, 5]
n1 = [1, 16]
n2 = [1, 16]
n3 = [1, 16]
n4 = [1, 16]
n5 = [1, 16]
activation = ['tanh', 'relu', 'sigmoid']
rho = [0.985, 0.995]
eps = [1e-9, 1e-7]
input_dropout_ratio = [0, 0.8]
l1 = [0.0, 1e-3]
l2 = [0.0, 1e-3]


def create_individual():

    g1 = random.choice(init_weight_distribution)
    g2 = random.choice(
        list(range(int(init_weight_scale[0]*10), int(init_weight_scale[1]*10)+1)))/10
    g3 = random.randrange(n_layer[0], n_layer[1]+1)
    g4 = random.randrange(n1[0], n1[1]+1)
    g5 = random.randrange(n2[0], n2[1]+1)
    g6 = random.randrange(n3[0], n3[1]+1)
    g7 = random.randrange(n4[0], n4[1]+1)
    g8 = random.randrange(n5[0], n5[1]+1)
    g9 = random.choice(activation)
    g10 = random.uniform(rho[0], rho[1])
    g11 = random.uniform(eps[0], eps[1])
    g12 = random.choice(
        list(range(int(input_dropout_ratio[0]*10), int(input_dropout_ratio[1]*10) + 1)))/10
    g13 = random.uniform(l1[0], l1[1])
    g14 = random.uniform(l2[0], l2[1])
    ind = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14]
    return ind


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate,
                 creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def custom_mutation(x, mpb):

    ind = deepcopy(x)

    if random.random() < mpb:
        mutation_point = random.choice(range(len(ind)))
        match mutation_point:
            case 0:
                ind[mutation_point] = random.choice(init_weight_distribution)
            case 1:
                ind[mutation_point] = random.choice(
                    list(range(int(init_weight_scale[0]*10), int(init_weight_scale[1]*10)+1)))/10
            case 2:
                ind[mutation_point] = random.randrange(
                    n_layer[0], n_layer[1]+1)
            case 3:
                ind[mutation_point] = random.randrange(n1[0], n1[1]+1)
            case 4:
                ind[mutation_point] = random.randrange(n2[0], n2[1]+1)
            case 5:
                ind[mutation_point] = random.randrange(n3[0], n3[1]+1)
            case 6:
                ind[mutation_point] = random.randrange(n4[0], n4[1]+1)
            case 7:
                ind[mutation_point] = random.randrange(n5[0], n5[1]+1)
            case 8:
                ind[mutation_point] = random.choice(activation)
            case 9:
                ind[mutation_point] = random.uniform(rho[0], rho[1])
            case 10:
                ind[mutation_point] = random.uniform(eps[0], eps[1])
            case 11:
                ind[mutation_point] = random.choice(
                    list(range(int(input_dropout_ratio[0]*10), int(input_dropout_ratio[1]*10) + 1)))/10
            case 12:
                ind[mutation_point] = random.uniform(l1[0], l1[1])
            case 13:
                ind[mutation_point] = random.uniform(l2[0], l2[1])

    return ind


def custom_crossover(ind1, ind2, cxpb):
    if random.random() < cxpb:
        c1, c2 = tools.cxUniform(ind1[:], ind2[:], 0.5)
        return creator.Individual(c1), creator.Individual(c2)
    else:
        return creator.Individual(ind1[:]), creator.Individual(ind2[:])


def evaluate_fitness(ind):
    fitness_value = get_accuracy(ind=ind)
    return (fitness_value,)


toolbox.register("select", tools.selTournament, tournsize=tournament_size)

toolbox.register("evaluate", evaluate_fitness)

toolbox.register("mate", custom_crossover, cxpb=p_cross)

toolbox.register("mutate", custom_mutation, mpb=p_mutation)


acc_per_run = []
all_configs = []
for _ in range(n_run):

    population = toolbox.population(n=population_size)
    fitnesses = list(map(toolbox.evaluate, population))

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for current_gen in range(n_generation):
        print(f'{current_gen=}')
        fitnesses = [ind.fitness.values[0] for ind in population]

        all_configs.extend([conf+[fit]
                           for conf, fit in zip(population, fitnesses)])
        print(f'{population[np.argmax(fitnesses)]=}\n{np.max(fitnesses)=}')
        offspring = []

        while len(offspring) < population_size:

            ind1, ind2 = random.sample(population, 2)
            child1, child2 = toolbox.mate(ind1, ind2)

            child1 = toolbox.mutate(child1)
            child2 = toolbox.mutate(child2)

            child1.fitness.values = evaluate_fitness(child1)
            child2.fitness.values = evaluate_fitness(child2)
            offspring.extend([child1, child2])

        population = toolbox.select(population + offspring, population_size)
    acc_per_run.append(max([ind.fitness.values[0] for ind in population]))


print(f'{np.mean(acc_per_run)=},{np.var(acc_per_run)}')
df = pd.DataFrame(data=all_configs, columns=['init_weight_distribution', 'init_weight_scale', 'n_layer',
                  'n1', 'n2', 'n3', 'n4', 'n5', 'activation', 'rho', 'eps', 'input_dropout_ratio', 'l1', 'l2', 'acc'])

df.to_csv('allconfigs.csv', index=False)
