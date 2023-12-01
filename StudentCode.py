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
import random

def generate_new_layout_swap_two(layout):
    """
    Generate a new layout by swapping two elements of the layout
    """
    new_layout = list(layout)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    new_layout[i], new_layout[j] = new_layout[j], new_layout[i]
    return new_layout

def generate_new_layout_from_parents(layout1, layout2):
    """
    Generate a new layout by generating a new permutation of the two parents
    """
    new_layout = list(layout1)
    for i in range(n):
        if np.random.rand() < 0.5:
            new_layout[i] = layout2[i] if layout2[i] not in new_layout else layout1[i]
    return new_layout
    
layout = list(range(n))
fitness(layout)

print(f"n={n}, m={m} et fitness_test={fitness(layout)}. Instance {instance_num} ok !")

random.shuffle(layout)
fitness(layout)
print(f"n={n}, m={m} et fitness_test={fitness(layout)}. Instance {instance_num} ok !")

def hill_climbing(layout):
    best_fitness = fitness(layout)
    best_layout = layout
    for i in range(10):
        new_layout = generate_new_layout_swap_two(layout)
        new_fitness = fitness(new_layout)
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            best_layout = new_layout
    return best_layout

best_layout = hill_climbing(layout)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")

"""
Current best results :
[19, 9, 16, 10, 4, 1, 5, 14, 12, 7, 15, 3, 11, 2, 0, 18, 13, 8, 17, 6]
n=20, m=27 et fitness_test=63. Instance 1 ok !
"""
def simulated_annealing(layout1, layout2, T=1):
    fitness1 = fitness(layout1)
    fitness2 = fitness(layout2)
    best_fitness = min(fitness1, fitness2)
    best_layout = layout1 if fitness1 < fitness2 else layout2
    for i in range(10000):
        new_layout1 = generate_new_layout_swap_two(layout1)
        new_layout2 = generate_new_layout_swap_two(layout2)
        new_fitness1 = fitness(new_layout1)
        new_fitness2 = fitness(new_layout2)
        if min(new_fitness1, new_fitness2) < best_fitness:
            best_fitness = min(new_fitness1, new_fitness2)
            best_layout = new_layout1 if new_fitness1 < new_fitness2 else new_layout2

        # Metropolis-Hastings (see : https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
        elif np.random.rand() < np.exp((best_fitness - max(new_fitness1, new_fitness2)) / T):
            best_fitness = max(new_fitness1, new_fitness2)
            best_layout = new_layout2 if new_fitness1 < new_fitness2 else new_layout1
    return best_layout

layout1 = np.random.permutation(n)
layout2 = np.random.permutation(n)
best_layout = simulated_annealing(layout1, layout2)
print(f"{best_layout}")
print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")

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

