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
        bestValue = recursive(listOfValues, listOfWeights, capacity, args.totalItems)
        end = time.time()
        print("Tempo em segundos", end - start)
        print("Solução:", bestValue)
    elif args.algorithm == "dynamic":
        start = time.time()
        bestValue = computeKnapsackProblemDynamicProgramming(listOfValues, listOfWeights, capacity, args.totalItems)
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
        #print(tuple, len(tuple), type(tuple))

        #soma pesos e valor
        sumWeights = 0
        sumValues = 0
        for elementID in tuple:
            sumWeights += listOfWeights[elementID]
            sumValues += listOfValues[elementID]
            #print(elementID, type(elementID))

        if sumWeights <= capacity:
            if sumValues > bestValue:
                #print(sumValues, sumWeights, tuple)
                bestValue = sumValues
                bestTuple = tuple

    return bestValue, bestTuple

def computeKnapsackProblemRecursiveMethod(listOfValues, listOfWeights, capacity, n):
    if n == 0:
        return 0
    elif listOfWeights[n - 1] > capacity:
        return recursive(listOfValues, listOfWeights, capacity, n - 1)
    else:
        return max([
            listOfValues[n - 1] + recursive(listOfValues, listOfWeights, capacity - listOfWeights[n - 1], n - 1), recursive(listOfValues, listOfWeights, capacity, n - 1)
        ])

def computeKnapsackProblemDynamicProgramming(listValor, listPeso, capacidadeMochila, n):
    K = np.array([[0 for x in range(capacidadeMochila + 1)] for x in range(n + 1)])
    print(K.shape)
    input()

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]

    return K[n][W]

#python main.py dumb 24 -w [382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684] -v [825594,1677009,1676628,1523970,943972,97426,69666,1296457,1679693,1902996,1844992,1049289,1252836,1319836,953277,2067538,675367,853655,1826027,65731,901489,577243,466257,369261] -W 6404180
#python main.py recursive 24 -w [382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684] -v [825594,1677009,1676628,1523970,943972,97426,69666,1296457,1679693,1902996,1844992,1049289,1252836,1319836,953277,2067538,675367,853655,1826027,65731,901489,577243,466257,369261] -W 6404180

#python main.py dumb 10 -w [23,31,29,44,53,38,63,85,89,82] -v [92,57,49,68,60,43,67,84,87,72] -W 165
#python main.py recursive 10 -w [23,31,29,44,53,38,63,85,89,82] -v [92,57,49,68,60,43,67,84,87,72] -W 165

#python main.py dumb 3 -v [60,100,120] -w [10,20,30] -W 50
#python main.py recursive 3 -v [60,100,120] -w [10,20,30] -W 50
#python main.py dynamic 3 -v [60,100,120] -w [1,2,3] -W 5
if __name__ == "__main__":
    main()