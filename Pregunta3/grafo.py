
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 23:13:04 2022

@author: YMAC
"""

import pandas as pd
import numpy as np
import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

#lectura del archivo csv
df = pd.read_csv("costos_.csv")
print("GRAFO")
print(df)
#Convirtiendo el df a matriz
matriz = np.array(df)
#print(matriz)
av = {
"Lugares" : len(matriz),#Se obtiene la cantidad de lugares que se visitará
"Matriz_costos" : matriz
}
costos = av["Matriz_costos"]
cant_lugares = av["Lugares"]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(cant_lugares), cant_lugares) 
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
ind = toolbox.individual() # creamos un individuo aleatorio

pop = toolbox.population(n=200) # creamos una población aleatoria

def evalAV(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""
    # distancia entre el último elemento y el primero
    distancia = costos[individual[-1]][individual[0]]
    
    # distancia entre el resto de ciudades
    #print("ind",individual, "dis ",distancia," - ", individual[-1], " - ",individual[0] )
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):        
        distancia += costos[gene1][gene2]
        #print("d :: ",distancia, gene1 , "-", gene2)
    return distancia,

toolbox.register("evaluate", evalAV)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=cant_lugares)

def main():
    random.seed(64) # ajuste de la semilla del generador de números aleatorios
    pop = toolbox.population(n=200) # creamos la población inicial
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    log = tools.Logbook()
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=False)
    return pop, hof, log

if __name__ == "__main__":
    pop, hof, log = main()
    road = pd.DataFrame(hof)
    road.replace(to_replace = 0 , value = "a", inplace = True)
    road.replace(to_replace = 1 , value = "b", inplace = True)
    road.replace(to_replace = 2 , value = "c", inplace = True)
    road.replace(to_replace = 3 , value = "d", inplace = True)
    road.replace(to_replace = 4 , value = "e", inplace = True)
    #print(log)
    print("Best cost: %f" %hof[0].fitness.values)
    #print("Tour:",road.iloc[0])
    print("Best way : ")
    print(road.iloc[0])
    
