# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
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
import threading


layout = list(range(n))


np.random.shuffle(layout)

def hill_climbing(layout, neigborhood_generation_method, size_tabu_list=10, nb_it=10):
    best_fitness = fitness(layout)
    best_layout = layout
    tabu_list = [layout]
    best_neighbor = best_layout
    for _ in range(nb_it):
        neighborhood = gen_neighborhood(best_neighbor, neigborhood_generation_method)
        best_neighbor = best_layout
        best_neighbor_fitness = fitness(best_neighbor)

        for neighbor in neighborhood:
            candidate_fitness = fitness(neighbor)
            if candidate_fitness < best_neighbor_fitness and neighbor not in tabu_list:
                best_neighbor = neighbor
                best_neighbor_fitness = candidate_fitness

        if best_fitness > best_neighbor_fitness:
            best_layout = best_neighbor
            best_fitness = best_neighbor_fitness
        
        append(tabu_list, best_neighbor, size_tabu_list)
   
    return best_layout



def append(tabu_list, element, size):
    if len(tabu_list) == size:
        tabu_list.pop(0)
    tabu_list.append(element)

def swap(l, i, j):
    temp = l[i]
    l[i] = l[j]
    l[j] = temp


def gen_neighborhood(layout, func):
    neighborhood = list(filter(lambda x : x != None, func(layout)))
    neighborhood.sort(key=lambda x : fitness(x))
    return neighborhood

def gen_neighborhood_inversion(layout):
    neighborhood = [None] * len(layout)

    for i in range(1, len(layout)):
        curr_neighbor = copy.copy(layout)
        swap(curr_neighbor, i, i - 1)
        neighborhood[i] = curr_neighbor

    return neighborhood

def gen_neighborhood_transposition(layout):
    neighborhood = [None] * len(layout)

    for i in range(len(layout)):
        for j in range(len(layout)):
            curr_neighbor = copy.copy(layout)
            swap(curr_neighbor, i, j)
            neighborhood[i] = curr_neighbor

    return neighborhood

def gen_neighborhood_moving(layout):
    neighborhood = [None] * len(layout)

    for i in range(len(layout)):
        for j in range(len(layout)):
            curr_neighbor = copy.copy(layout)
            element = curr_neighbor.pop(j)
            curr_neighbor.insert(i, element)
            neighborhood[i] = curr_neighbor

    return neighborhood

def thread_task(len_list, layout):
    best_layout = hill_climbing(layout, gen_neighborhood_inversion, len_list)
    print(f"{best_layout}")
    print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")
    print(f"length of the tabu list : {len_list}")

def test_length(max_len_list): 
    layout = list(range(n))
    np.random.shuffle(layout)
    threads = [None] * (max_len_list - 2) 
    for i in range(0, max_len_list - 2):
        t_layout = copy.copy(layout)
        threads[i] = threading.Thread(target=thread_task, args=(i + 2, t_layout))
        threads[i].start()
    
    [threads[i].join() for i in range(0,max_len_list - 2)]

"""
best_layout = hill_climbing(layout, gen_neighborhood_moving)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")
best_layout = hill_climbing(layout, gen_neighborhood_inversion)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")
best_layout = hill_climbing(layout, gen_neighborhood_transposition)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")
"""
test_length(10)

#best_layout = test(layout)


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

