# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import numpy as np
from multiprocessing import Process
from numpy._typing import NDArray
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeSingaporeV2,FakeWashingtonV2,FakeCairoV2

##-------------------------------------------------------
##     Definition de la fonction objetif à minimiser
##-------------------------------------------------------
def fitness(layout) -> int:
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
    elif instance_num==1:
        return "Cairo","ghz",19
    else:
        print("Choix d'une instance inexistance, instance 1 revoyé  par défaut")
        return "Cairo","ghzall",20


##-------------------------------------------------------
##     Pour choisir une instance: 
##     Modifier instance_num ET RIEN D'AUTRE    
##-------------------------------------------------------
instance_num=1     #### Entre 1 et 9 inclue

backend_name,circuit_type,num_qubit=instance_selection(instance_num)
backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

n=num_qubit
m=backend.num_qubits

##-------------------------------------------------------
##     A vous de jouer !  
##-------------------------------------------------------

###### Votre code ici

# Depuis ce code, on sait que la solution est de la forme [0,1,2,...,n-1].
# On peut donc tester la fonction fitness sur cette solution et optimiser son resultat.
# La metaheuristique ne doit se baser que sur le layout et la fonction fitness.
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

import time
import numpy as np
from multiprocessing import Process, Lock, Array


mutex = Lock()

def gen_neighborhood(n, a, ith_permutation)->None:
    """
    will generate the ith permutation from an @initial_solution.

    This function use the iterative heap algorithm for more space efficiency.
    I FORGOT THAT NUMPY HAS ALREADY ONE METHOD FOR THIS FUUUUUCK
    """
    c = [0] * n
    i = 1
    nb_perm = 0
    while i < n:
        if nb_perm == ith_permutation:
            return a

        if c[i] < i:
            if i & 1:
                a[c[i]], a[i] = a[i], a[c[i]]
            else:
                a[0], a[i] = a[i], a[0]

            nb_perm += 1
            c[i] += 1
            i = 1
        else:
            c[i] = 0
            i += 1


def process_task(neighborhood, i, best_solution_so_far):
    neighbor = gen_neighborhood(n, np.copy(neighborhood), np.random.randint(n, size=1)) #shake
    print(f"process {i} has found the neighbor : {neighbor}")
    print(f"started local search...")
    current_solution = simulated_annealing(neighbor)
    current_solution_cost = fitness(current_solution)

    neighborhood_cost = fitness(neighborhood)

    if current_solution_cost < neighborhood_cost:
        print(f"best neighbor found for process {i} ! from {neighborhood}, {fitness(neighborhood)} to {current_solution}, {current_solution_cost}")
        for i in range(n):
            neighborhood[i] = current_solution[i]
        neighborhood_cost = current_solution_cost
    
    with mutex:
        print("je passe")
        if neighborhood_cost < fitness(best_solution_so_far): #update best sol if better sol
            print(f"best solution changed ! from {fitness(best_solution_so_far)} -> {neighborhood_cost} ")
            for i in range(n):
                best_solution_so_far[i] = neighborhood[i]

        

def vns(m, n, number_of_neighborhood=5, max_time=120):
    """
    Variable neighborhood search in wich the simulated annealing is used for local search
    #TODO make on different processes
    """
    st = time.time()

    neighborhood_set = np.array([np.random.RandomState().choice(m, n, replace=False) for _ in range(number_of_neighborhood)])    
    shared_neighborhood_set = [Array('i', range(n))] * number_of_neighborhood
    for i in range(number_of_neighborhood):
        for j in range(n):
            shared_neighborhood_set[i][j] = neighborhood_set[i][j]
    best_solution_so_far = neighborhood_set[0]
    shared_best_solution = Array('i', range(n))
    for i in range(n):
        shared_best_solution[i] = best_solution_so_far[i]
    
    while (time.time() - st) < max_time:
        print(f"one big iteration, curr_time : {time.time() - st}")

        p = [Process(target=process_task, args=(shared_neighborhood_set[i], i, shared_best_solution)) for i in range(number_of_neighborhood)]

        for i in range(number_of_neighborhood):
            print(f"Started process {i} with the current neighborhood : {neighborhood_set[i]}")
            p[i].start()
        
        [p[i].join() for i in range(number_of_neighborhood)]

        best_solution_so_far = np.frombuffer(shared_best_solution.get_obj(), dtype=np.int32)
        print(f"#################the best soltion found {best_solution_so_far} ########################")

    return best_solution_so_far


def simulated_annealing(initial_solution, initial_temperature : float = 100, minimal_temperature : float = 0.1, step : int = 5, coef : float = 0.8):
    best_solution_so_far = initial_solution
    fitness_best_solution = fitness(best_solution_so_far)

    while initial_temperature > minimal_temperature:
        
        for _ in range(step):
    
            neighbor = np.random.permutation(best_solution_so_far)
            fitness_neighbor = fitness(neighbor)
            delta_E = fitness_neighbor - fitness_best_solution
            
            if delta_E <= 0 or np.exp(delta_E/initial_temperature) > np.random.uniform(0, 1):
                best_solution_so_far = neighbor
                fitness_best_solution = fitness_neighbor
        
        initial_temperature *= coef

    return best_solution_so_far

def main():
    NB_PROCESS = 8
    MAX_TIME=120
    """ 
    for i in range(1,10):
        instance_num = i
        print(f"-----------------------{i}TH INSTANCE-----------------------")
        backend_name,circuit_type,num_qubit=instance_selection(instance_num)
        backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

        n=num_qubit
        m=backend.num_qubits
        solution_found = vns(m, n, NB_PROCESS, MAX_TIME)
        print(f"BEST SOLUTION FOUND FOR INSTANCE {i} : {solution_found}")

    """
    solution_found = vns(m, n, NB_PROCESS, MAX_TIME)
    print(f"BEST SOLUTION FOUND FOR INSTANCE 1 : {solution_found}")



if __name__ == "__main__":
    main()

