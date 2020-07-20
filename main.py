from argparse import ArgumentParser
import sys
import re
import random
import numpy as np

def parse_args():
    parser = ArgumentParser(description = 'Knapsack Problem Framework')
    parser.add_argument('algorithm', help = 'Algoritmos possíveis (dumb, recursive, dynamic, genetic)')
    parser.add_argument('totalItems', type = int, help = 'Total de Itens')
    parser.add_argument('-seed', dest = 'seed',  type = int, required = False, help = 'Semente')
    parser.add_argument('-v', action = 'store', dest = 'values', required = False,
                        help = 'Vetor com valores de cada item entre colchetes e com virgula (ex: [60,100,120])')
    parser.add_argument('-w', action = 'store', dest = 'weights', required = False,
                        help = 'Vetor com pesos de cada item entre colchetes e com virgula (ex: [1,2,3])')
    parser.add_argument('-W', action = 'store', dest = 'W', type = int, required = False,
                        help = 'Capacidade da mochila')

    parser.add_argument('-sp', action = 'store', dest = 'sizeOfPopulation', type = int, required = False,
                        help = 'Tamanho da população (para algoritmo genético)')
    parser.add_argument('-ng', action = 'store', dest = 'numGenerations', type = int, required = False,
                        help = 'Número de gerações (para algoritmo genético)')
    parser.add_argument('-mr', action = 'store', dest = 'mutationRate', type = float, required = False,
                        help = 'Taxa de Mutação (para algoritmo genético)')

    return parser.parse_args()

def main():
    args = parse_args()

    if(args.values is not None and args.weights is not None and args.W is not None):
        listOfValues = [int(s) for s in re.findall(r'\b\d+\b', args.values)]
        listOfWeights = [int(s) for s in re.findall(r'\b\d+\b', args.weights)]

        if len(listOfValues) != args.totalItems or len(listOfWeights) != args.totalItems:
            print("ERRO: Listas devem ter o mesmo número de elementos iguais ao número de itens passados (totalItems)")
            return

        if args.W <= 0:
            print("ERRO: Capacidade da mochila deve ser maior do que zero")
            return

        capacity = args.W
    else:
        if args.seed is not None:
            random.seed(args.seed)
        listOfValues = random.sample(range(1, 9999), int(args.totalItems))
        if args.seed is not None:
            random.seed(args.seed)
        listOfWeights = random.sample(range(1, 9999), int(args.totalItems))

        if args.W is not None:
            capacity = args.W
        else:
            if args.seed is not None:
                random.seed(args.seed)
            capacity = random.randint(1, 99999)

    import time
    if args.algorithm == "dumb":
        start = time.time()
        bestValue, bestTuple = computeKnapsackProblemDumbMethod(listOfValues, listOfWeights, capacity)
        end = time.time()
        print("Tempo em segundos", end - start)
        print("Solução:", bestValue, bestTuple)
    elif args.algorithm == "recursive":
        start = time.time()
        bestValue = computeKnapsackProblemRecursiveMethod(listOfValues, listOfWeights, capacity, args.totalItems)
        end = time.time()
        print("Tempo em segundos", end - start)
        print("Solução:", bestValue)
    elif args.algorithm == "dynamic":
        start = time.time()
        bestValue = computeKnapsackProblemDynamicProgramming(listOfValues, listOfWeights, capacity, args.totalItems)
        end = time.time()
        print("Tempo em segundos", end - start)
        print("Solução:", bestValue)
    elif args.algorithm == "genetic":
        start = time.time()
        bestValue = computeKnapsackProblemGeneticAlgorithm(listOfValues, listOfWeights, capacity, args.totalItems, args.sizeOfPopulation, args.numGenerations, args.mutationRate, args.seed)
        end = time.time()
        print("Tempo em segundos", end - start)
        print("Solução:", bestValue)

def computeKnapsackProblemDumbMethod(listOfValues, listOfWeights, capacity):
    import itertools as it

    listOfItemsById = []
    for i in range(0, len(listOfValues)):
        listOfItemsById.append(i)

    allCombinations = []
    for i in range(1, len(listOfItemsById) + 1):
        auxComb = list(it.combinations(listOfItemsById, i))
        for a in auxComb:
            allCombinations.append(a)

    bestValue = -1
    bestTuple = None
    for tuple in allCombinations:
        #soma pesos e valor
        sumWeights = 0
        sumValues = 0
        for elementID in tuple:
            sumWeights += listOfWeights[elementID]
            sumValues += listOfValues[elementID]

        if sumWeights <= capacity:
            if sumValues > bestValue:
                bestValue = sumValues
                bestTuple = tuple

    return bestValue, bestTuple

def computeKnapsackProblemRecursiveMethod(listOfValues, listOfWeights, capacity, n):
    if n == 0:
        return 0
    elif listOfWeights[n - 1] > capacity:
        return computeKnapsackProblemRecursiveMethod(listOfValues, listOfWeights, capacity, n - 1)
    else:
        return max([
            listOfValues[n - 1] + computeKnapsackProblemRecursiveMethod(listOfValues, listOfWeights, capacity - listOfWeights[n - 1], n - 1), computeKnapsackProblemRecursiveMethod(listOfValues, listOfWeights, capacity, n - 1)
        ])

def computeKnapsackProblemDynamicProgramming(listOfValues, listOfWeights, maxCapacity, n):
    K = np.array([[0 for x in range(maxCapacity + 1)] for x in range(n + 1)])

    for it in range(n + 1):
        for cap in range(maxCapacity + 1):
            if it == 0 or cap == 0:
                K[it][cap] = 0
            elif listOfWeights[it - 1] <= cap:
                K[it][cap] = max(listOfValues[it - 1] + K[it - 1][cap - listOfWeights[it - 1]], K[it - 1][cap])
            else:
                K[it][cap] = K[it - 1][cap]

    return K[n][maxCapacity]

import matplotlib.pyplot as plt
def computeFitnessOfPopulation(listOfValues, listOfWeights, listOfPopulation, maxCapacity, numberOfItems, sizeOfPopulation):
    fitness = []
    for i in range(sizeOfPopulation):
        valueOfSolutionI = np.sum(listOfPopulation[i] * listOfValues)
        weightOfSolutionI = np.sum(listOfPopulation[i] * listOfWeights)

        if weightOfSolutionI <= maxCapacity:
            fitness = np.append(fitness, valueOfSolutionI)
        else:
            fitness = np.append(fitness, 0)

    return fitness

def getFittestIndividuals(fitnessOfPopulation, numberOfParents, listOfPopulation):
    fitnessOfIndividualsSorted = list(fitnessOfPopulation)
    fitnessOfIndividualsSorted.sort(reverse = True)

    #[120, 100, 60, 0]

    fittestIndividualsIndices = []
    while 1:
        indexOfIndividualsWithThatFitness = np.where(fitnessOfIndividualsSorted[len(fittestIndividualsIndices)] == fitnessOfPopulation)
        for eachFittestIndividualIndex in indexOfIndividualsWithThatFitness[0]:
            fittestIndividualsIndices = np.append(fittestIndividualsIndices, eachFittestIndividualIndex)

            if numberOfParents == len(fittestIndividualsIndices):
                parents = np.zeros((numberOfParents, listOfPopulation.shape[1]), dtype = int) #linha, coluna
                row = 0
                for eachIndividualIndex in fittestIndividualsIndices:
                    parents[row, :] = listOfPopulation[int(eachIndividualIndex)]
                    row += 1

                return parents

def generateChildByOnePointCrossover(parents, numberOfChildren):
    children = np.zeros((numberOfChildren, parents.shape[1]), dtype = int) #linha, coluna

    selectedParents = random.sample(list(parents), numberOfChildren)
    for i in range(0, numberOfChildren, 2):
        crossoverPoint = random.randint(1, parents.shape[1] - 1)

        randomParent1 = selectedParents[i]
        randomParent2 = selectedParents[i + 1]

        children[i, 0:crossoverPoint] = randomParent1[:crossoverPoint]
        children[i, crossoverPoint:] = randomParent2[crossoverPoint:]

        children[i + 1, 0:crossoverPoint] = randomParent2[:crossoverPoint]
        children[i + 1, crossoverPoint:] = randomParent1[crossoverPoint:]

    return children

def mutateIndividuals(individuals, mutationRate):
    for individual in individuals:
        randomValue = random.random()
        if randomValue <= mutationRate:
            sortedGene = random.randint(0, individual.shape[0] - 1)
            if individual[sortedGene] == 0:
                individual[sortedGene] = 1
            else:
                individual[sortedGene] = 0

def computeKnapsackProblemGeneticAlgorithm(listOfValues, listOfWeights, maxCapacity, n, sizeOfPopulation, numGenerations, mutationRate, seed):
    #https://medium.com/koderunners/genetic-algorithm-part-3-knapsack-problem-b59035ddd1d6

    if sizeOfPopulation is None:
        print("ERRO: Faltou informar o tamanho da população no algoritmo genético.")
        exit()

    if numGenerations is None:
        print("ERRO: Faltou informar o número de gerações no algoritmo genético.")
        exit()

    if mutationRate is None:
        print("ERRO: Faltou informar a taxa de mutação no algoritmo genético.")
        exit()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    listOfPopulation = np.random.randint(2, size=(sizeOfPopulation, n))

    fitnessHistory = []
    for g in range(numGenerations):
        fitnessForPopulation = computeFitnessOfPopulation(listOfValues, listOfWeights, listOfPopulation, maxCapacity, n, sizeOfPopulation)
        fitnessHistory.append(fitnessForPopulation)

        numberOfParents = int(sizeOfPopulation / 2)
        parents = getFittestIndividuals(fitnessForPopulation, numberOfParents, listOfPopulation)
        children = generateChildByOnePointCrossover(parents, numberOfParents)
        mutateIndividuals(children, mutationRate)

        listOfPopulation[0 : parents.shape[0], :] = parents
        listOfPopulation[parents.shape[0] :, :] = children

    finalFitness = computeFitnessOfPopulation(listOfValues, listOfWeights, listOfPopulation, maxCapacity, n, sizeOfPopulation)

    print(listOfPopulation)
    print(finalFitness)

    fitnessHistoryMean = [np.mean(f) for f in fitnessHistory]
    fitnessHistoryMax = [np.max(f) for f in fitnessHistory]

    print(fitnessHistoryMean)
    print(fitnessHistoryMax)

    '''
    plt.plot(list(range(numGenerations)), fitnessHistoryMean, label = 'Fitness Médio')
    plt.plot(list(range(numGenerations)), fitnessHistoryMax, label = 'Fitness Máximo')
    plt.legend()
    plt.title('Fitness ao longo das gerações')
    plt.xlabel('Gerações')
    plt.ylabel('Fitness')
    plt.show()
    '''

    return max(finalFitness)

#python main.py dumb 24 -w [382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684] -v [825594,1677009,1676628,1523970,943972,97426,69666,1296457,1679693,1902996,1844992,1049289,1252836,1319836,953277,2067538,675367,853655,1826027,65731,901489,577243,466257,369261] -W 6404180
#python main.py recursive 24 -w [382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684] -v [825594,1677009,1676628,1523970,943972,97426,69666,1296457,1679693,1902996,1844992,1049289,1252836,1319836,953277,2067538,675367,853655,1826027,65731,901489,577243,466257,369261] -W 6404180

#python main.py dumb 10 -w [23,31,29,44,53,38,63,85,89,82] -v [92,57,49,68,60,43,67,84,87,72] -W 165
#python main.py recursive 10 -w [23,31,29,44,53,38,63,85,89,82] -v [92,57,49,68,60,43,67,84,87,72] -W 165

#python main.py dumb 3 -v [60,100,120] -w [10,20,30] -W 50
#python main.py recursive 3 -v [60,100,120] -w [10,20,30] -W 50
#python main.py dynamic 3 -v [60,100,120] -w [1,2,3] -W 5
if __name__ == "__main__":
    main()
