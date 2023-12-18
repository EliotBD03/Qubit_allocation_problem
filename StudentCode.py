# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import numpy as np
import time
import sys
import threading
from multiprocessing import Process, Queue, Pool, cpu_count

from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeSingaporeV2,FakeWashingtonV2,FakeCairoV2



#global stopFirst
#stopFirst = True


class NewThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)


    def run(self):
        
        if self._target != None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

class NewProcess(Process):
    def __init__(self,queue: Queue, args = (), functToCall = None):
        super().__init__()
        self.queue = queue
        self.args = args
        self.funct = functToCall

    def run(self) -> None:
        res = ()
        if(self.funct != None):
            res = self.funct(self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],False)
        else:
            res = Local_SearchOnRange(self.args[0],self.args[1], True)
        self.queue.put(res)
    




##-------------------------------------------------------
##     Definition de la fonction objetif à minimiser
##-------------------------------------------------------

def fitness_thread(layout) -> tuple:
    return (layout, fitness(layout))

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

def RVNS(m:int, n:int, neighborhoods: list, maxTime= 100000, nbOfThreads=9, allowPanic = True):
    print("Starting RVNS with " + str(nbOfThreads) +" threads allowed")
    local_random_seed = np.random.RandomState()
    x = local_random_seed.choice(m,n,replace=False)
    
    #stuckSince = 0
    #lastUnstuck = 0
    #divCount = 0
    #maxDiv = 1
    notMoved = 0
    base_Panic = 15 
    current_PanicDetector = base_Panic
    #fx = fitness(x)
    fx = float('inf')
    real_x, real_fx = x,fx
    print("Starting with x= " + str(x))
    # Idea, use several thread to find a better solution for the others when stuck !! *Emergency threads*    
    print("fx = " + str(fx))
    startTime = time.time()
    while(time.time()-startTime < maxTime):
        k = 0
        if allowPanic and notMoved >= current_PanicDetector:
                notMoved = 0
                if not current_PanicDetector <= 10:        
                    current_PanicDetector = current_PanicDetector // 2
                print("=-" * 100)
                print("Start Panic/Diversification Mode")
                (x,fx) = DiversificationRVNS(m,n,neighborhoods,40,3)
                print("Stop Diversification")
                print("=-"*100)
                print(x,fx)
                if(fx < real_fx):
                    current_PanicDetector = base_Panic
                    real_x,real_fx = x,fx
                    print("_" * 20)
                    print("New Solution:")
                    print("x = " + str(x))
                    print("fx = " + str(fx))

        while(k < len(neighborhoods) and time.time() - startTime < maxTime):
            

            #stuckSince = time.time() -startTime - lastUnstuck
            #print("Stuck since: " + str(stuckSince))
            print("Time spent: " + str(time.time() - startTime) + " sec")
            print("k = :" + str(k))
            #print("__" * 10)
            #print(" Shake Solution")
            s = ShakeSol(x, neighborhoods[k])
            s,f_s = Local_SearchWithProcess(neighborhoods[k], s, len(s), nbOfThreads)

            if(f_s < fx):
                fx = f_s
                x = s
                k = 0
                notMoved = 0
                if(fx < real_fx):
                    current_PanicDetector = base_Panic
                    real_x,real_fx = x,fx
                    print("_" * 20)
                    print("New Solution:")
                    print("x = " + str(x))
                    print("fx = " + str(fx))
            else:
                notMoved += 1
                k += 1
    return (real_x, real_fx)


def DiversificationRVNS(m:int, n:int, neighborhoods: list, maxTime= 100, nbOfThreads=4):
    startTime = time.time()
    while time.time() - startTime < maxTime:
        

        Q = Queue()
        processes = []
        for i in range(nbOfThreads):
            p = NewProcess(Q, functToCall=RVNS, args=(m,n,neighborhoods,maxTime/4,nbOfThreads))
            processes.append(p)
            p.start()
        currMin = float('inf')
        currBestList = []
        for i in range(nbOfThreads):
            curr = Q.get()
        
            if (curr[1] < currMin):
                currMin = curr[1]
                currBestList = curr[0]
    return (currBestList, currMin)


def SVNS(n: int, neighborhoods: list, maxTime: (15 * 60), alpha: int):
    x = np.random.choice(m,n,replace=False)
    fx = float('inf')
    #fx = fitness(x)
    #print("Starting RVNS")
    startTime = time.time()
    #x,fx = RVNS(n,neighborhoods, maxTime/5)
    print("Starting SVNS")
    print("Starting with x= " + str(x))
    
    print("fx = " + str(fx))
    real_x,real_fx = x,fx
    
    while(time.time() - startTime < maxTime):
        k = 0
        while k < len(neighborhoods) and time.time() - startTime < maxTime:
            print("k: " + str(k))
            s = ShakeSol(x,neighborhoods[k])
            #best_s,f_bs = Local_Search(neighborhoods[k], s, len(s), True)
            best_s,f_bs = Local_SearchWithProcess(neighborhoods[k], s, len(s), 15)
            print(best_s,f_bs)
            #best_s,f_bs = s, fitness(s)
            if( f_bs - (alpha * distance(x,best_s)) < fx ):
            #if( f_bs - (alpha * np.abs(f_bs - fx)) < fx ):
                x = best_s
                fx = f_bs
                if(fx < real_fx):
                    k = 0
                    real_x,real_fx = x,fx
                    print("New Best Sol:")
                    print("x = " + str(x))
                    print("fx = " + str(fx))
                    
            else:
                k+= 1
            

        
        x,fx = real_x, real_fx
    if(fx < real_fx):
        real_x,real_fx = x,fx
        print("New Best Sol:")
        print("x = " + str(x))
        print("fx = " + str(fx))        

    return (real_x, real_fx)



def VNS_Real(Size: int, nList: list, maxTime = 100):
    i = 0
    startTime = time.time()
    nSize = len(nList)
    currIt = 0
    #print("GRASP starts")
    #best = GRASP(size, 3)
    print("VNS starts")
    s = np.random.choise(m, size, replace=False)
    best = (s,float('inf'))
    shakedSol = []
    bestShakedSol = []
    while i < nSize and (time.time() - startTime) <= maxTime:
        print("New iteration")
        shakedSol = ShakeSol(best[0], nList[i], size)
        bestShakedSol = Local_SearchWithProcess(nList[i], shakedSol, size,2)
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

def nextAddNeighbor(l,n):
    nextList = list(copy(l))
    
    for i in range(1,m):
        for j in range(len(l)):
            nextList[j] = (nextList[j] + i) % m 
        yield nextList


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
    # This is bad but nuh huh
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

def Local_SearchWithThread(neighborhood, sol: list, size:int, nbOfThreads: int):
    curr = 0
    threads = []
    currBestList = []
    currMin = float('inf')
    # Get all neighbors
    neighbors = [copy(n) for n in neighborhood(sol, size)]
    threadRange = int(len(neighbors) / nbOfThreads)
    for i in range(nbOfThreads):
        print("Started Thread: " + str(i))
        t = NewThread(target=Local_SearchOnRange,args=(neighbors[i*threadRange:i*threadRange + threadRange],sol))
        threads.append(t)
        t.start()
    for i in range(nbOfThreads):
        curr = threads[i].join()
        if (curr[1] < currMin):
            currMin = curr[1]
            currBestList = curr[0]
    return (currBestList, currMin) 


def Local_SearchWithProcess(neighborhood, sol: list, size:int, nbOfProcess: int):
    curr = float('inf')
    processes = []
    currBestList = []
    currMin = float('inf')
    queue = Queue()
    # Get all neighbors
    neighbors = [copy(n) for n in neighborhood(sol, size)]
    # nbOfProcess + 1, because we can also use the main processus 
    threadRange = int(len(neighbors) / (nbOfProcess + 1))
    i = 0
    while i < nbOfProcess-1:
        #print("Started Process: " + str(i))
        #p = NewThread(target=Local_SearchOnRange,args=(neighbors[i*threadRange:i*threadRange + threadRange],sol))
        p = NewProcess(queue=queue,args=(neighbors[i*threadRange:i*threadRange + threadRange],sol))
        processes.append(p)
        p.start()
        i += 1
    p = NewProcess(queue=queue,args=(neighbors[i*threadRange:],sol))
    processes.append(p)
    p.start()

    i = 0
    while i < nbOfProcess:
        curr = queue.get()
        
        i += 1
        
        if (curr[1] < currMin):
            currMin = curr[1]
            currBestList = curr[0]
    return (currBestList, currMin) 





def Local_SearchOnRange(neighborhoodList: list, sol: list, firstBestResult= False):
    curr= 0
    currBestList = sol
    currMin = fitness(sol)
    for i in range(len(neighborhoodList)):
        
        curr = fitness(neighborhoodList[i])
        
        if(curr < currMin):

            currBestList = neighborhoodList[i]
            currMin = curr
            if(firstBestResult):
                return (currBestList,curr)
    
    return (currBestList,currMin)
        

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
nbOfInstances = 10
nbOfThreads = cpu_count()
res = []



nbOfMinutes = 10
maxTime = ((nbOfMinutes)*60)/(nbOfInstances-1)
print("Allowing " + str(maxTime) + " seconds for each instances")

"""
Start = time.time()
print("Without Threads:")
#print(Local_Search(nextPermutationNeighbor,perm,n))
print("It took: "+ str(time.time() - Start )  + "seconds")
Start = time.time()
print("With Threads:")
print(Local_SearchWithProcess(nextPermutationNeighbor,perm,n,nbOfThreads))
print("It took: "+ str(time.time() - Start)  + "seconds")
"""


for i in range(1,nbOfInstances):
    
    # Removes the randomness, for testing purpuses
    #np.random.seed(0)
    # Max 15 min -> 10 instance : (15* 60)/10 for each
    instance_num= i    #### Entre 1 et 9 inclue
    print("_-" * 36)
    print("-_"*15 + " INSTANCE: " + str(instance_num) +"-_" * 15)
    print("_-" * 36)
    
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)
    f = open("./results/SVNS_1", "a")

    n=num_qubit
    global m 
    m = backend.num_qubits
    alpha = 8
    res = RVNS(m,n,[nextPermutationNeighbor,nextInversionNeighbor,nextAddNeighbor],maxTime)
    #res = SVNS(n,[nextInversionNeighbor,nextInversionNeighbor,nextAddNeighbor],maxTime,alpha)
    f.write(("-=" * 10) + "Instances " + str(instance_num) + ("=-" * 10) + "\n")
    f.write(str(res) + "\n")
    
    print(res)
    #res.append(RVNS(m,n,[nextInversionNeighbor,nextPermutationNeighbor,nextAddNeighbor],maxTime))
    #print("Alpha= " + str(alpha))
    f.close()
    #res.append(SVNS(n,[nextAddNeighbor,nextInversionNeighbor,nextPermutationNeighbor],maxTime, alpha))
    



""" 1800 sec
1: 47 | 41
2: 46 | 57
3: 67 | 59
4: 68 | 78
5: 57 | 75      
6: 90 | 83
7: 23 | 38      (1500 sec)
8: 30 | 52
9: 43 | 53


7:
x = [ 4  3  0 14 26 18  2 23 21 17 15 24  1  6  7  8 12 16 10 19 22 20 13  5
 25  9 11]
fx = 76
x = [ 4  3 14  0 26 18  2 23 21  5 24  1 17  6  8  7 12 16 10 19 22 20 13 15
 25  9 11]
fx = 67
x = [ 4  3 21  0 26 18  2 23 14  5 15  1  6  7 17  8 12 16 19 10 22 20 13 24
 25  9 11]
fx = 61
[ 4 20  3  0 25 18  2 23 14  5  1 15  6  7 17  8 12 16 19 21 22 13 10 24
 26  9 11]
With a cost of: 52


x = [ 8 11  7 13 10  6  9 12  5 17 16  3 19 18  0  1 14  2  4 15]
fx = 38



"""
