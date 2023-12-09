# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import numpy as np
import time
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

def VNS_FunnyVersion(n:int, neighborhoods: list, maxTime= float('inf')):
    x = np.random.permutation(n)
    fx = fitness(x)
    currOpt = (x[:],fx)
    k = 0
    startTime = time.time()

    while (k < len(neighborhoods) and (time.time()-startTime) < maxTime):
        print(time.time()-startTime)
        print("k = " + str(k))
        s = ShakeSol(x, neighborhoods[k], n) # x'
        # Local_Search
        print("Start Local_Search") 
        bestShakedSol = Local_Search(neighborhoods[k],s, n)  # x''
        #print(bestShakedSol)
        print("Finished Local Search")
        if bestShakedSol[1] < currOpt[1]: # f(x'') < f_opt
            currOpt = (bestShakedSol[0],fx) # f_opt = f(x) et x_opt = x''
        
        print(distance(bestShakedSol[0], x))
        if(bestShakedSol[1] - (10* distance(bestShakedSol[0],x))) < fx or ((bestShakedSol[0] == x).all()):
            print("Accepted")
            x = bestShakedSol[0]
            fx = bestShakedSol[1]
            k = 0
        else:
            print("Not  accepted")
            k += 1
        
        print("currOpt: " + str(currOpt))
        print("x: " + str(x) + " | f(x): " + str(fx))
    return (x, fx)

def distance(perm1: list, perm2: list) -> int:
    res = 0
    for i in range(len(perm1)):
        if(perm1[i] != perm2[i]):
            res += 1
    return res

def RVNS(n:int, neighborhoods: list, maxTime= float('inf')):
    x = np.random.permutation(n)
    fx = fitness(x)
    print("Starting with x= " + str(x))
    
    print("fx = " + str(fx))
    startTime = time.time()
    while(time.time()-startTime < maxTime):
        k = 0
        while(k < len(neighborhoods)):
            #print("__" * 10)
            #print(" Shake Solution")
            s = ShakeSol(x, neighborhoods[k])
            s,f_s = Local_Search(neighborhoods[k], s, len(s), True)

            #print(" Get Fitness for: "  + str(s))
            #f_s = fitness(s)
            if(f_s < fx):
                fx = f_s
                x = s
                k = 0
                print("New Solution:")
                print("x = " + str(x))
                print("fx = " + str(fx))
            else:
                k += 1
    return (x, fx)


def SVNS(n: int, neighborhoods: list, maxTime: (15 * 60), alpha: int):
    x = np.random.permutation(n)
    fx = fitness(x)

    real_x,real_fx = x,fx
    
    startTime = time.time()
    while(time.time() - startTime < maxTime):
        k = 0
        while k < len(neighborhoods) and time.time() - startTime < maxTime:
            s = ShakeSol(x,neighborhoods[k])
            best_s,f_bs = Local_Search(neighborhoods[k], s, len(s), True)
            if( f_bs - (alpha * distance(x,best_s)) < fx ):
                x = best_s
                fx = f_bs
                k = 0
            else:
                k+= 1

        if(fx < real_fx):
            real_x,real_fx = x,fx
            print("New Best Sol:")
            print("x = " + str(x))
            print("fx = " + str(fx))
        x,fx = real_x, real_fx
            

    return (real_x, real_fx)



def VNS_Real(size: int, nList: list, maxIt = 5):
    i = 0
    currIt = 1
    nSize = len(nList)
    print("GRASP starts")
    best = GRASP(size, maxIt/2)
    print("VNS starts")
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
            maxIt -= 1
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


def ShakeSol(s, neighborhood, size = -1):
    if (size == -1):
        size = len(s)
    neighbors = [copy(n) for n in neighborhood(s, size)]
    return neighbors[np.random.randint(0,len(neighbors)-1)]

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
    # This is bad but huh
    return np.random.permutation(size)


def Local_Search(neighborhood, sol: list,size: int, firstImprovement = False):
    curr = 0
    currBestList = []
    currMin = float('inf')
    for neighbor in neighborhood(sol,size):
            curr = fitness(neighbor)
            if curr < currMin:
                currMin = curr
                currBestList = copy(neighbor)
                if(firstImprovement):
                    break
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

nbOfMinutes = 15
maxTime = (nbOfMinutes*60)/10
print("Allowing " + str(maxTime) + " seconds for each instances")
for i in range(1,10):
    # Max 15 min -> 10 instance : (15* 60)/10 for each
    print("-_" * 32)
    print("-_"*15 + " INSTANCE: " + str(i) +"-_" * 15)
    print("-_" * 32)
    instance_num= i    #### Entre 1 et 9 inclue
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

    n=num_qubit
    m=backend.num_qubits
    #res.append(copy(VNS_Real(n, [nextInversionNeighbor, nextPermutationNeighbor])))
    #res.append(copy(RVNS(n, [nextPermutationNeighbor,nextInversionNeighbor],maxTime)))
    res.append(copy(SVNS(n, [nextPermutationNeighbor,nextInversionNeighbor],maxTime,4)))
    
print("=_=" * 20)
for i in range(len(res)):
    print("Solution for each instances : " + str((i)*2))
    print(res[i][0])
    print("With a cost of:")
    print(res[i][1])
    print("____________________________")

