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
instance_num=8     #### Entre 1 et 9 inclue

backend_name,circuit_type,num_qubit=instance_selection(instance_num)
backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

n=num_qubit
m=backend.num_qubits

##-------------------------------------------------------
##     Definition de la fonction objetif à minimiser
##-------------------------------------------------------
def fitness(layout, qr=qr, qc=qc, backend=backend) -> int:
    init_layout={qr[i]:layout[i] for i in range(len(layout))}
    init_layout=Layout(init_layout)

    pm = generate_preset_pass_manager(3,backend,initial_layout=init_layout)
    pm.layout.remove(1)
    pm.layout.remove(1)

    QC=pm.run(deepcopy(qc))
    return QC.depth()

##-------------------------------------------------------
##     A vous de jouer !  
##-------------------------------------------------------

###### Votre code ici

# Depuis ce code, on sait que la solution est de la forme [0,1,2,...,n-1].
# On peut donc tester la fonction fitness sur cette solution et optimiser son resultat.
# La metaheuristique ne doit se baser que sur le layout et la fonction fitness.
import time
from copy import deepcopy


def generate_new_layout_swap_two(layout):
    """
    Generate a new layout by swapping two elements of the layout
    """
    i, j = np.random.choice(len(layout), size=2, replace=False)
    new_layout = layout.copy()
    new_layout[i], new_layout[j] = new_layout[j], new_layout[i]
    return new_layout

def generate_new_layout_random(layout, m):
    """
    Generate a new layout by generating a new random layout
    """
    new_layout = np.random.choice(m, len(layout), replace=False)
    return new_layout

    
layout = np.random.choice(m, n, replace=False)
fitness(layout)

print(f"n={n}, m={m} et fitness_test={fitness(layout)}. Instance {instance_num} ok !")

# np.random.shuffle(layout)
# fitness(layout)
# print(f"n={n}, m={m} et fitness_test={fitness(layout)}. Instance {instance_num} ok !")
#
# def hill_climbing(layout):
#     best_fitness = fitness(layout)
#     best_layout = layout
#     for i in range(10):
#         new_layout = generate_new_layout_swap_two(layout)
#         new_fitness = fitness(new_layout)
#         if new_fitness < best_fitness:
#             best_fitness = new_fitness
#             best_layout = new_layout
#     return best_layout
#
# best_layout = hill_climbing(layout)
# print(f"{best_layout}")
# print(f"n={n}, m={m} et fitness_test={fitness(best_layout)}. Instance {instance_num} ok !")

"""
Current best results :
[19, 9, 16, 10, 4, 1, 5, 14, 12, 7, 15, 3, 11, 2, 0, 18, 13, 8, 17, 6]
n=20, m=27 et fitness_test=63. Instance 1 ok !
"""
def simulated_annealing(layout, T=1, alpha=0.99, time_limit=10, return_results=[], qr=qr, qc=qc, backend=backend, _type=1, instance=1, m=m):
    """
    Simulated annealing algorithm for the layout problem

    Parameters
    ----------
    layout : list
        The layout to optimize.
    T : float, optional
        The initial temperature. The default is 1.
    alpha : float, optional
        The cooling factor. The default is 0.99.
    time_limit : float, optional
        The time limit for the algorithm. The default is 10.
    return_results : list, optional
        The list in which to append the results. The default is [].
    qr : list, optional
        The list of qubits. The default is qr.
    qc : list, optional
        The quantum circuit. The default is qc.
    backend : list, optional
        The backend. The default is backend.
    type : int, optional
        The type of simulated annealing to use. The default is 1 (swap two elements, intensification). Other values are 2 (generate a new random layout, diversification).
    """
    print(f"Starting simulated annealing... Type : {_type}. Instance {instance}")
    _b_time = time.time()
    layout1 = layout.copy()
    T_copy = deepcopy(T)
    fitness1 = fitness(layout1, qr=qr, qc=qc, backend=backend)
    best_fitness = fitness1
    best_layout = layout1
    bests_list = [(best_layout, best_fitness)]

    rand_values = np.random.rand(int(time_limit * 10)).tolist()  # Precompute random values

    print(f"Simulated annealing starting for instance {instance}...")

    while time.time() - _b_time < time_limit:
        # Generate a new layout
        match _type:
            case 1:
                new_layout1 = generate_new_layout_swap_two(layout1)
            case 2:
                new_layout1 = generate_new_layout_random(layout1, m)
            case _:
                new_layout1 = generate_new_layout_swap_two(layout1)

        new_fitness1 = fitness(new_layout1, qr=qr, qc=qc, backend=backend)

        if new_fitness1 < fitness1 or rand_values.pop() < np.exp((fitness1 - new_fitness1) / T_copy):
            layout1, fitness1 = new_layout1, new_fitness1
            bests_list.append((layout1, fitness1))
            if fitness1 < best_fitness:
                best_fitness, best_layout = fitness1, layout1

        T_copy *= alpha

    if return_results is not None:
        return_results.extend(bests_list)

    print(f"Simulated annealing done for instance {instance} !")

    return bests_list


def simulated_annealing_iteration(layout, iterations, T, alpha, qr, qc, backend):
    """
    Simulated annealing algorithm for the layout problem with a fixed number of iterations
    """
    layout1 = layout.copy()
    fitness1 = fitness(layout1, qr, qc, backend)
    best_fitness = fitness1
    best_layout = layout1
    rand_values = np.random.rand(iterations).tolist()  # Precompute random values
    for _ in range(iterations):
        new_layout1 = generate_new_layout_swap_two(layout1)
        new_fitness1 = fitness(new_layout1, qr, qc, backend)
        if new_fitness1 < fitness1 or rand_values.pop() < np.exp((fitness1 - new_fitness1) / T):
            with open(f"./outputs2/output_{instance_num}.txt", "a") as file:
                file.write(f"{new_layout1}\n")
                file.write(f"n={n}, m={m} et fitness_test={new_fitness1}. Instance {instance_num} !\n----------------\n")
            layout1, fitness1 = new_layout1, new_fitness1           
            if fitness1 < best_fitness:
                best_fitness, best_layout = fitness1, layout1
        T *= alpha
    return (best_layout, best_fitness)


target_fitnesses = [46, 45, 66, 67, 55, 89, 22, 29, 42]
def multi_thread_simulated_annealing(layout, ex, reduction, thread_number=10, instance=1):
    """
    Multi thread simulated annealing algorithm for the layout problem.
    The algorithm is based on the Cluster Algorithm for Simulated Annealing (CA) which can be found in the following paper :
    https://www.sciencedirect.com/science/article/pii/S0743731596901215

    Parameters
    ----------
    layout : list
        The layout to optimize.
    ex : float
        The exchange parameter for partial results (temperature).
    reduction : float
        The reduction parameter for the temperature.
    thread_number : int, optional
        The number of threads (workers) to use. The default is 10.
    instance : int, optional
        The instance number. The default is 1.
    """
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

    n=num_qubit
    m=backend.num_qubits

    # Get the 'thread_number' initial layouts for the threads
    layouts = [layout.copy() for _ in range(thread_number)]
    for i in range(thread_number):
        np.random.shuffle(layouts[i])

    # Get the target fitness for the threads
    target_fitness = target_fitnesses[instance - 1]

    # Get the initial fitnesses for the threads
    fitnesses = [fitness(layouts[i], deepcopy(qr), deepcopy(qc), deepcopy(backend)) for i in range(thread_number)]

    # Initialize the workers
    workers = [threading.Thread(target=_multi_thread_simulated_annealing, args=(layouts[i], ex, reduction, fitnesses[i], target_fitness, deepcopy(qr), deepcopy(qc), deepcopy(backend))) for i in range(thread_number)]

    print(f"Starting simulated annealing for instance {instance}...")
    
    # Start the workers
    for worker in workers:
        worker.start()

    # Wait for the workers to finish
    for worker in workers:
        worker.join()

    # Get the best layout and fitness
    best_layout, best_fitness = min(zip(layouts, fitnesses), key=lambda x: x[1])


    print(f"Best layout found : {best_layout}")
    print(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance} !")

    with open(f"./outputs2/output_{instance_num}.txt", "a") as file:
        file.write(f"\n\nBest layout found :\n")
        file.write(f"{best_layout}\n")
        file.write(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance_num} !\n----------------\n")


def _multi_thread_simulated_annealing(layout, ex, reduction, current_fitness, target_fitness, qr, qc, backend):
    """
    Simulated annealing algorithm for the layout problem.
    The algorithm is based on the Cluster Algorithm for Simulated Annealing (CA) which can be found in the following paper :
    https://www.sciencedirect.com/science/article/pii/S0743731596901215

    Parameters
    ----------
    layout : list
        The layout to optimize.
    ex : float
        The exchange parameter for partial results (temperature).
    reduction : float
        The reduction parameter for the temperature.
    current_fitness : int
        The current fitness of the layout.
    target_fitness : int
        The target fitness of the layout.
    """
    while current_fitness > target_fitness and ex > 0.01:
        # Generate a new layout
        results = simulated_annealing_iteration(layout, 10, ex, 0.99, qr, qc, backend)

        # Get the new layout and fitness
        new_layout, new_fitness = results

        # If the new layout is better, update the current layout and fitness
        if new_fitness < current_fitness:
            layout, current_fitness = new_layout, new_fitness

        ex -= reduction


# layout1 = np.random.permutation(n)
# bests = simulated_annealing(layout1, T=10, alpha=0.99, time_limit=100)
# print(f"Best layouts found after diversification :")
# for best_layout, best_fitness in bests:
#     print(f"{best_layout}")
#     print(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance_num} !")

import threading

# Multi thread simulated annealing intensification for all the best layouts found
def multi_thread_simulated_annealing_intensification(bests, T=1, alpha=0.99, time_limit=100, n=n, file=None, qc=qc, qr=qr, backend=backend, instance=instance_num):
    threads = []
    results = [[] for _ in range(5)]
    bests_five = deepcopy(bests) 
    bests_five.sort(key=lambda x: x[1])
    bests_five = bests_five[:5]
    print(f"bests_five : {bests_five}")
    if file is not None:
        file.write(f"\n\nStarting threads for intensification of 5 bests...\n")
    print("Starting threads...")
    for i in range(min(5, len(bests))):
        threads.append(threading.Thread(target=simulated_annealing, args=(bests_five[i][0], T, alpha, time_limit, results[i], qr, qc, backend, 1, instance, m)))
        threads[-1].start()
        
    for thread in threads:
        thread.join()

    # Print the best layout found by each thread
    if file is not None:
        file.write(f"\nBest layouts found after intensification :\n")
    print("Best layouts found after intensification :")
    end_results = []
    for i in range(len(results)):
        end_results += results[i]

    for (best_layout, best_fitness) in end_results:
        if file is not None:
            file.write(f"{best_layout}\n")
            file.write(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance} !\n----------------\n")

        print(f"{best_layout}")
        print(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance} !")

    end_results.sort(key=lambda x: x[1])
    if file is not None:
        file.write(f"\nFinal best layout and cost : {min(end_results, key=lambda x: x[0][1])}")
    print(f"Final best layout and cost : {min(end_results, key=lambda x: x[0][1])}")
#multi_thread_simulated_annealing_intensification(bests, T=1, alpha=0.8, time_limit=100)


def run_all_instances():
    threads = []
    for i in range(1, 10):
        threads.append(threading.Thread(target=run_instance, args=(i,)))
        threads[-1].start()
    for thread in threads:
        thread.join()

def run_instance(instance_num):
    print(f"Running instance {instance_num}...")
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

    n=num_qubit
    m=backend.num_qubits

    layout = np.random.choice(m, n, replace=False)

    file = open(f"./outputs/output_{instance_num}.txt", "w")
    f1 = fitness(layout, qr=qr, qc=qc, backend=backend)
    file.write(f"Random : f{layout}\n")
    file.write(f"n={n}, m={m} et fitness_test={f1}. Instance {instance_num} ok !\n----------------\n")

    file.write(f"Beginning diversification for instance {instance_num}...\n")
    print(f"Beginning diversification for instance {instance_num}...")
    print("Starting threads...")
    bests = [[] for _ in range(2)]
    threads = []
    for i in range(2):
        threads.append(threading.Thread(target=simulated_annealing, args=(layout, 100, 0.9, 1000, bests[i], qr, qc, backend, 2, instance_num, m)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    
    # Merge the results of the threads
    bests = bests[0] + bests[1]

    print(f"Diversification done for instance {instance_num} !")
    file.write(f"\n\nBest layouts found after diversification :\n")
    for (best_layout, best_fitness) in bests:
        file.write(f"{best_layout}\n")
        file.write(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance_num} !\n----------------\n")

    bests.sort(key=lambda x: x[1])

    best_layouts = bests[:3] + bests[-2:] # Keep the 3 bests and the 2 worsts

    bests_intensification = [[] for _ in range(min(5, len(best_layouts)))]
    threads = []
    file.write(f"\n\nStarting threads for intensification of 5 bests...\n")
    print("Starting threads...")
    file.write(f"\n5 bests : {best_layouts}\n")
    for i in range(min(5, len(best_layouts))):
        # Optimize the best layouts found with simulated annealing
        threads.append(threading.Thread(target=simulated_annealing, args=(best_layouts[i][0], 1, 0.9, 1000, bests_intensification[i], qr, qc, backend, 1, instance_num, m)))
        threads[-1].start()
    for thread in threads:
        thread.join()

    # Merge the results of the threads
    i_results = []
    for i in range(len(bests_intensification)):
        i_results += bests_intensification[i]

    print(f"Intensification done for instance {instance_num} !")
    file.write(f"\n\nBest layouts found after intensification :\n")
    for (best_layout, best_fitness) in i_results:
        file.write(f"{best_layout}\n")
        file.write(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance_num} !\n----------------\n")

    print(f"Final best layout and cost : {min(min(i_results, key=lambda x: x[1])[1], min(bests, key=lambda x: x[1])[1])}")

    file.write(f"\nFinal best cost : {min(min(i_results, key=lambda x: x[1])[1], min(bests, key=lambda x: x[1])[1])}")

    file.close()



    # for instance_num in range(1, 10):
    #     backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    #     backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)
    #
    #     n=num_qubit
    #     m=backend.num_qubits
    #
    #     layout = np.random.permutation(n)
    #
    #
    #     file = open(f"./outputs/output_{instance_num}.txt", "w")
    #
    #     f1 = fitness(layout, qr=qr, qc=qc, backend=backend)
    #     file.write(f"Random : f{layout}\n")
    #     file.write(f"n={n}, m={m} et fitness_test={f1}. Instance {instance_num} ok !\n----------------\n")
    #     print(f"n={n}, m={m} et fitness_test={f1}. Instance {instance_num} ok !")
    #
    #     bests = simulated_annealing(layout, T=10, alpha=0.99, time_limit=20, qr=qr, qc=qc, backend=backend, n=n)
    #     file.write(f"\n\nBest layouts found after diversification :\n")
    #     print(f"Best layouts found after diversification :")
    #     for best_layout, best_fitness in bests:
    #         print(f"{best_layout}")
    #         file.write(f"{best_layout}\n")
    #         print(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance_num} !")
    #         file.write(f"n={n}, m={m} et fitness_test={best_fitness}. Instance {instance_num} !\n----------------\n")
    #
    #     file.close()

#run_all_instances()

for instance_num in range(1, 10):
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

    n=num_qubit
    layout = np.random.permutation(n)

    multi_thread_simulated_annealing(layout, 100, 0.9, 8, instance=instance_num)


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

