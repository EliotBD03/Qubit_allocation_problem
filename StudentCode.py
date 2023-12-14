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


def layout_swap(layout):
    new_layout = layout.copy()
    i = np.random.randint(0, len(new_layout))
    j = np.random.randint(0, len(new_layout))
    new_layout[i], new_layout[j] = new_layout[j], new_layout[i]
    return new_layout


def aco(layout, num_ants=10, num_iterations=100, rho=0.1, alpha=1.0, beta=2.0):
    """
    num_ants: nombre de fourmis
    num_iterations: nombre d'Itérations
    rho: taux d'évaporation des phéromones
    alpha: influence de la qualité de la solution
    beta: influence des phéromones
    """
    ants = [layout_swap(layout) for _ in range(num_ants)]

    best_solution = None
    best_fitness = np.inf

    for iteration in range(num_iterations):
        fitness_values = [fitness(lay) for lay in ants]
        print(fitness_values)

        pheromones = np.ones((n, n))
        selected_layouts = []
        for ant_index in range(num_ants):
            probabilities = [(pheromones[ants[ant_index][i-1]][j] ** alpha) * (1.0 / fitness_values[ant_index] ** beta) for i, j in enumerate(ants[ant_index])]
            probabilities /= sum(probabilities)
            selected_layouts.append(ants[ant_index][np.random.choice(n, p=probabilities)])

        # Evaporation phéromones
        for i in range(n):
            for j in range(n):
                pheromones[i][j] = (1 - rho) * pheromones[i][j]

        # for ant_index in range(num_ants):
        #     for i in range(n - 1):
        #         pheromones[selected_layouts[ant_index][i]][selected_layouts[ant_index][i+1]] += 1 / fitness_values[ant_index]
        
        if min(fitness_values) < best_fitness:
            best_solution = ants[fitness_values.index(min(fitness_values))]
            best_fitness = min(fitness_values)

    print("Meilleure solution:", best_solution)
    print("Fitness de la meilleure solution:", best_fitness)

layout = np.random.permutation(n)
aco(layout, num_ants=10, num_iterations=100, rho=0.1, alpha=1.0, beta=2.0)
