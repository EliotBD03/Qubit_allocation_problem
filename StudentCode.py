# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import copy
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
instance_num=3     #### Entre 1 et 9 inclue

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
import random

layout = list(range(n))


random.shuffle(layout)

def hill_climbing(layout, neigborhood_generation_method, nb_it=10, size_tabu_list=10):
    best_fitness = fitness(layout)
    best_layout = layout
    tabu_list = [layout] + [None] * (size_tabu_list - 1)
    for i in range(nb_it):
        best_neighbor = find_best_neighbor(gen_neighborhood_for_tabu_search(best_layout, tabu_list, neigborhood_generation_method))
        fitness_best_neighbor = fitness(best_neighbor)
        
        if best_fitness == fitness_best_neighbor:
            return best_layout #local minimal reached
        best_layout = best_neighbor
        best_fitness = fitness_best_neighbor
        append(tabu_list, best_neighbor)
    return best_layout



def append(tabu_list, element):
    if tabu_list[-1] != None:
        tabu_list = tabu_list[1:]
    tabu_list.append(element)

def swap(l, i, j):
    temp = l[i]
    l[i] = l[j]
    l[j] = temp

def find_best_neighbor(neighborhood):
    fitness_neighborhood = list(map(fitness, neighborhood))
    min_i = 0
    min_fitness = fitness(neighborhood[0])
    for i in range(1,len(fitness_neighborhood)):
        if min_fitness > fitness_neighborhood[i]:
            min_i = i
            min_fitness = fitness_neighborhood[i]
    return neighborhood[min_i]

def gen_neighborhood_for_tabu_search(layout, tabu_list, func):
    return [neighbor for neighbor in func(layout) if not neighbor in tabu_list]

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


def destroy_my_pc(layout):
    tabu_size_list = 5
    nb_it = 2
    curr_methods = [gen_neighborhood_moving, gen_neighborhood_transposition, gen_neighborhood_inversion]
    curr_sols = [None] * 3
    best_solution = layout
    for i in range(1,tabu_size_list):
        for j in range(1,nb_it):
            for k, curr_method in enumerate(curr_methods):
                curr = hill_climbing(layout, curr_method, j, i)
                print(k)
                curr_sols[k] = [curr, fitness(curr)]
            best_curr_sol = 0
            for l in range(1,len(curr_sols)):
                if fitness(curr_sols[best_curr_sol]) > fitness(curr_sols[l]):
                    best_curr_sol = l 
            if fitness(best_solution) > fitness(curr_sols[best_curr_sol]):
                best_solution = curr_sols[best_curr_sol]
                print(f"ne solution found ! {best_solution} {fitness(best_solution)} with it = {j} and tabu_size_list = {i}")


"""
def hill_climbing_thread(layout, neigborhood_generation_method, buff, nb_it=10, size_tabu_list = 10):
    best_fitness = fitness(layout)
    best_layout = layout
    tabu_list = [layout] + [None] * (size_tabu_list - 1)
    for i in range(nb_it):
        
        best_neighbor = find_best_neighbor(gen_neighborhood_for_tabu_search(best_layout, tabu_list, neigborhood_generation_method))
        fitness_best_neighbor = fitness(best_neighbor)
        
        if best_fitness == fitness_best_neighbor:
            return best_layout #local minimal reached
        elif best_fitness > fitness_best_neighbor:
            best_layout = best_neighbor
            best_fitness = fitness_best_neighbor
        append(tabu_list, best_neighbor)

    buff = best_layout


def test(layout):
    best_solution = layout
    current_solutions = [None] * 3
    methods = [gen_neighborhood_inversion, gen_neighborhood_moving, gen_neighborhood_transposition]
    for i, method in enumerate(methods):
        for k in range(10):
            thread = Thread(target=hill_climbing, args=(best_solution, method, current_solutions[i]))
            thread.start()
            for j in range(len(methods)):
                if current_solutions[j] != None and fitness(best_solution) > fitness(current_solutions[j]):
                    best_solution = current_solutions[j]
    return best_solution   
"""

#destroy_my_pc(layout)


best_layout = hill_climbing(layout, gen_neighborhood_moving)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")
best_layout = hill_climbing(layout, gen_neighborhood_inversion)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")
best_layout = hill_climbing(layout, gen_neighborhood_transposition)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")



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

