# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeSingaporeV2,FakeWashingtonV2,FakeCairoV2

##-------------------------------------------------------
##     Definition de la fonction objetif à minimiser
##-------------------------------------------------------
def fitness(layout) -> list:
    init_layout={qr[i]:layout[i] for i in range(len(layout))}
    init_layout=Layout(init_layout)

    pm = generate_preset_pass_manager(3,backend,initial_layout=init_layout)
    pm.layout.remove(1)
    pm.layout.remove(1)

    QC=pm.run(qc)
    return QC.depth()

##-------------------------------------------------------
##     Selection de l'instance du probleme
##-------------------------------------------------------
def instance_characteristic(backend_name,circuit_type,num_qubit):
    if backend_name == "Singapore":
        backend = FakeSingaporeV2()
    elif backend_name == "Cairo":
        backend = FakeCairoV2()
    else :
        backend = FakeWashingtonV2()
        
    l=f"{circuit_type}_indep_qiskit_{num_qubit}"
    qasmfile=f"./Instances/{l.rstrip()}.qasm"  ###### Il est possible de cette ligne soit problèmatique.
    qc=QuantumCircuit().from_qasm_file(qasmfile)
    qr=qc.qregs[0]
    
    return backend,qc,qr

def instance_selection(instance_num):
    if instance_num==1:
        return "Cairo","ghzall",20
    elif instance_num==2:
        return "Wash","ghzall",20
    elif instance_num==3:
        return "Cairo","ghzall",27
    elif instance_num==4:
        return "Wash","ghzall",27
    elif instance_num==5:
        return "Wash","dj",20
    elif instance_num==6:
        return "Cairo","dj",27
    elif instance_num==7:
        return "Wash","ghz",20
    elif instance_num==8:
        return "Wash","ghz",27    
    elif instance_num==9:
        return "Cairo","qaoa",14
    elif instance_num==11:
        return "Singapore","ghzall",19
    elif instance_num==12:
        return "Singapore","dj",19
    elif instance_num==13:
        return "Cairo","ghz",19
    else:
        print("Choix d'une instance inexistance, instance 1 revoyé  par défaut")
        return "Cairo","ghzall",20

##-------------------------------------------------------
##     A vous de jouer !  
##-------------------------------------------------------

###### Votre code ici

# Depuis ce code, on sait que la solution est de la forme [0,1,2,...,n-1].
# On peut donc tester la fonction fitness sur cette solution et optimiser son resultat.
# La metaheuristique ne doit se baser que sur le layout et la fonction fitness.
import random
from copy import copy

def VNS_AxelVersion(n:int):
    """
        Main method using the VNS method.

        Neighborhoods used on permutations:

                - Inversion
                - Permutation
                - Movement

        Parameters
        ----------
        :param n: int

        Returns
        -------
    """
    neighborList = [nextInversionNeighbor,
                    nextMovementNeighbor,
                    nextPermutationNeighbor
                    ]
    nbOfNeigh = len(neighborList)
    alreadyVisited = [[]] * nbOfNeigh

    currNeighFinished = 0
    currNeighb = 0
    changed = False
    
    currBestList = np.random.permutation(n+1)
    currMin = fitness(currBestList)
    # Initial Solution
    while currNeighFinished < nbOfNeigh:
        print(currNeighFinished)
        neighborhood = neighborList[(currNeighb + 1)%nbOfNeigh]
        skip = False
        try:
            alreadyVisited[(currNeighb + 1)%nbOfNeigh].index(currBestList)
        except ValueError:
            currNeighFinished += 1
            skip = True
        if skip:
            pass
            
        
        alreadyVisited[(currNeighb + 1)%nbOfNeigh].append(currBestList)
        currNeighb += 1
        changed = False
        bestNeighbor = Local_Search(neighborhood, currBestList, n)
        if bestNeighbor[1] < currMin:
            print("NEW BEST !!!\n From :" + str(currMin) + " to " + str(bestNeighbor[1]))    
            print("Previous: " + str(currBestList))
            print("New:      " + str(bestNeighbor[0]))
            print("______________________________________________")
            currBestList = bestNeighbor[0]
            currMin = bestNeighbor[1]

        if not changed:
            currNeighFinished += 1
        elif currNeighFinished > 0:
            currNeighFinished = 0

    return (currBestList, currMin)

def VNS_Real(size: int, nList: list, maxIt = 10):
    i = 0
    currIt = 1
    nSize = len(nList)

    best = GRASP(size, maxIt/2)

    #best = (s,fitness(s))
    shakedSol = []
    bestShakedSol = []
    while i < nSize and currIt <= maxIt:
        print("New iteration !\n    " + str(currIt))
        shakedSol = ShakeSol(best[0], nList[i], size)
        bestShakedSol = Local_Search(nList[i], shakedSol, size)
        if (bestShakedSol[1] < best[1] or (acceptWithError(best[1], bestShakedSol[1], currIt))):
        
            print("NEW Solution !!!\n From :" + str(best[1]) + " to " + str(bestShakedSol[1]))    
            print("Previous: " + str(best[0]))
            print("New:      " + str(bestShakedSol[0]))
            print("______________________________________________")
            best = bestShakedSol
            i = 0
        else:
            i += 1
        currIt += 1

    return best

def acceptWithError(currValue: int, solValue: int, t: int):
    if(currValue == solValue):
        return False
    prob = np.power(np.e,(currValue-solValue)/t)

    print(prob)
    if(random.random() < prob):
        print("Accepted With Error")
        return True
    return False


def ShakeSol(s, neighborhood, size):
    neighbors = [n[:] for n in neighborhood(s, size)]
    return neighbors[random.randint(0,len(neighbors)-1)]

def nextInversionNeighbor(l, n):
    nextList = copy(l)
    for i in range(0,n-1):
        swap(nextList,i,i+1)
        yield nextList
        swap(nextList, i+1,i)

def nextPermutationNeighbor(l,n):
    nextList = copy(l)
    for i in range(n) :
        for j in range(i+1,n):
            swap(nextList,i,j)
            yield nextList
            swap(nextList,j,i)

def nextMovementNeighbor(l,n):
    nextList = list(copy(l))
    for i in range(n-1, 0, -1):
        for j in range(n):
            curr = nextList.pop(i)
            curr = nextList.insert(j,curr)
            yield nextList
            nextList = list(copy(l))
            

def swap(l,i,j):
    l[i],l[j] = l[j], l[i]

#for v in nextInversionNeighbor(list(range(n)), 20):
#    print(v)
#________________________________________________________________________________________________________
def GRASP(size, maxIteration) -> tuple:

    BestSolution = ([],float('inf'))
    currSolution = ([],0)
    currIteration = 1
    while(maxIteration > 0):
        print("========================================================")
        print("New Iteration Of Grasp: " + str(currIteration))
        currIteration += 1
        print("========================================================")
        # Construction
        sol = Greedy_Randomized_Construction(size)

        # Local Search
        currSolution = Local_Search(nextInversionNeighbor,sol,size)

        # Update Solution
        if currSolution[1] < BestSolution[1]:
            print("NEW BEST !!!\n From :" + str(BestSolution[1]) + " to " + str(currSolution[1]))    
            print("Previous: " + str(BestSolution[0]))
            print("New: " + str(currSolution[1]))
            print("______________________________________________")
            BestSolution = currSolution

        maxIteration -= 1
    return BestSolution

def Greedy_Randomized_Construction(size):
    
    return random.sample(range(0,size), size)


def Local_Search(neighborhood, sol: list,size: int):
    curr = 0
    currBestList = []
    currMin = float('inf')
    for neighbor in neighborhood(sol,size):
            curr = fitness(neighbor)
            if curr < currMin:
                currMin = curr
                currBestList = copy(neighbor)
    #print("     Local search result: " + str(currMin))
    return (currBestList, currMin)


#for i in nextMovementNeighbor(list(range(4)), 4):
#    print(i)


###### A faire : un algo d'optimisation qui minimise la fonction fitness,
###### fonction qui accepte en entrée :
###### une liste de n parmi m (n<=m) entiers deux à deux distincts
###### N'oubliez pas d'écrire la solution dans [GROUPE]_instance_[instance_num].txt

###### /!\ Attention /!\
###### Il est possible que la ligne 40 : qasmfile=f".\Instances\{l.rstrip()}.qasm"
###### crée un problème à l'execution si le chemin n'est pas le bon,
###### en particulier sous Linux. Essayer de la remplacer par
###### qasmfile=f"./Instances/{l.rstrip()}.qasm". Cela devrait résoudre le problème.

###### Voici un test (à supprimer !) pour s'assurer que tout va bien
# for i in range(1,10):
#     instance_num=i     #### Entre 1 et 9 inclue
#
#     backend_name,circuit_type,num_qubit=instance_selection(instance_num)
#     backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)
#
#     print(f"Instance {instance_num} : {backend_name}, {circuit_type}, {num_qubit} qubits")
#     print(f"qc : {qc.draw()}")
#     print(f"qr : {qr}")
#
#     n=num_qubit
#     m=backend.num_qubits
#     r=fitness(list(range(n)))
#     print(f"n={n}, m={m} et fitness_test={r}. Instance {instance_num} ok !")
#
#
#



##-------------------------------------------------------
##     Pour choisir une instance: 
##     Modifier instance_num ET RIEN D'AUTRE    
##-------------------------------------------------------
res = []
for i in range(1,10):
    print("_-_-_-_-_-_-_-_-INSTANCE: " + str(i) +"-_-_-_-_-_-_-_-_-_-_-_-_-_-")
    instance_num= i+1     #### Entre 1 et 9 inclue
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

    n=num_qubit
    m=backend.num_qubits
    res.append(copy(VNS_Real(n, [nextInversionNeighbor, nextPermutationNeighbor])))


for i in range(len(res)):
    print("Solution for instance : " + str(i + 1))
    print(res[i][0])
    print("With a cost of:")
    print(res[i][1])
    print("____________________________")

