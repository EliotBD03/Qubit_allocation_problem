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

from multiprocessing import Process, active_children, Queue

import os

class NewProcess(Process):
    def __init__(self, queue: Queue, args = ()):
        super().__init__()
        self.daemon = True
        self.queue = queue
        self.args = args

    def run(self):
        try:
            res = simulated_annealing(*self.args)
            self.queue.put(res)
        except KeyboardInterrupt:
            pass


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
instance_num=7     #### Entre 1 et 9 inclue

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


def generate_new_layout_swap_two(layout, m=0, cur_layout=0):
    """
    Generate a new layout by swapping two elements of the layout
    """
    i, j = np.random.choice(len(layout), size=2, replace=False)
    new_layout = layout.copy()
    new_layout[i], new_layout[j] = new_layout[j], new_layout[i]
    return new_layout

def generate_new_layout_random(layout, m, cur_layout):
    """
    Generate a new layout by generating a new random layout from the current layout

    Parameters
    ----------
    layout : list
        The current layout.
    m : int
        The number of qubits.
    cur_layout : int
        The number of times n fits in m (used for better neighborhood generation).
    """
    new_layout = np.random.choice(m - (len(layout) * cur_layout), len(layout), replace=False)
    return new_layout

def generate_new_layout_add(layout, m, cur_layout):
    """
    Generate a new layout by adding a new element to the layout

    Parameters
    ----------
    layout : list
        The current layout.
    m : int
        The number of qubits.
    cur_layout : int
        The number of times n fits in m (used for better neighborhood generation).
    """
    new_layout = layout.copy()
    new_element = np.random.randint(m - (len(layout) * cur_layout))
    new_layout = list(map(lambda x: (x + new_element) % m, new_layout))
    return new_layout



def hill_climbing(layout, goal, qr, qc, backend, time_limit):
    """
    Hill climbing algorithm for the layout problem (local search)

    Parameters
    ----------
    layout : list
        The layout to optimize.
    goal : int
        The goal fitness.
    qr : list
        The list of qubits.
    qc : list
        The quantum circuit.
    backend : list
        The backend.
    time_limit : float
        The time limit for the algorithm.
    """
    best_fitness = fitness(layout, qr, qc, backend)
    best_layout = layout
    start_time = time.time()
    while best_fitness > goal and time.time() - start_time < time_limit:
        new_layout = generate_new_layout_swap_two(layout)
        new_fitness = fitness(new_layout, qr, qc, backend)
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            best_layout = new_layout
    return (best_layout, best_fitness)


def simulated_annealing(layout, T=1, alpha=0.99, time_limit=10, return_results=[], qr=qr, qc=qc, backend=backend, _type=1, instance=1, m=m, cur_layout=0, nbr_process=-1, path="./outputs2/"):
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
        The type of simulated annealing to use. The default is 1 (swap two elements, intensification). Other values are 2 (generate a new random layout, diversification) and 3 (generate new layout from adding a certain number to all the elements of the layout).
    instance : int, optional
        The instance number. The default is 1.
    m : int, optional
        The number of qubits. The default is m.
    cur_layout : int, optional
        The number of times n fits in m (used for better neighborhood generation). The default is 0.
    nbr_process : int, optional
        The number of processes to use. The default is -1 (all the processes).
    path : str, optional
        The path to the output file. The default is "./outputs2/".
    """
    print(f"Starting simulated annealing... Type : {_type}. Instance {instance}")
    # Set the seed for the current thread
    local_seed = np.random.RandomState()
    local_seed.seed()
    np.random.seed(local_seed.randint(0, 1000000000))

    # Initialize the parameters
    layout1 = layout.copy()
    T_copy = deepcopy(T)
    fitness1 = fitness(layout1, qr=qr, qc=qc, backend=backend)
    best_fitness = fitness1
    best_layout = layout1
    bests_list = [(best_layout, best_fitness)]

    rand_values = np.random.rand(int(time_limit * 100)).tolist()
    print(f"Simulated annealing starting for instance {instance}...")

    start_time = time.time()

    # Run the algorithm for 'time_limit' seconds (of thread actual running time)
    while time.time() - start_time < time_limit:
        # Generate a new layout from the selected type
        match _type:
            case 1:
                new_layout1 = generate_new_layout_swap_two(layout1)
            case 2:
                new_layout1 = generate_new_layout_random(layout1, m, cur_layout)
            case _:
                new_layout1 = generate_new_layout_add(layout1, m, cur_layout)

        new_fitness1 = fitness(new_layout1, qr=qr, qc=qc, backend=backend)

        # If the new layout is better, keep it, else keep it with a certain probability
        if new_fitness1 < fitness1 or rand_values.pop() < np.exp((fitness1 - new_fitness1) / T_copy):
            layout1, fitness1 = new_layout1, new_fitness1
            bests_list.append((layout1, fitness1))

            # Write the results to the output file if there is one
            if os.path.exists(path):
                with open(f"{path}output_{instance}.txt", "a") as file:
                    file.write(f"{layout1}\n")
                    file.write(f"n={n}, m={m} et fitness_test={fitness1}. Instance {instance} !\n----------------\n")

            if fitness1 < best_fitness:
                best_fitness, best_layout = fitness1, layout1

        T_copy *= alpha # cooling
        if T_copy == 0:
            # Restart the algorithm
            T_copy = T

    if return_results is not None:
        return_results.extend(bests_list)

    print(f"Simulated annealing done for instance {instance} !")

    # Keep the best results depending on the number of processes (threads)
    bests_list.sort(key=lambda x: x[1])

    return bests_list[:nbr_process]



def run_all_instances(time=100, path="./outputs2/"):
    """
    Run all the instances in parallel

    Parameters
    ----------
    time : float, optional
        The time limit for each instance. The default is 100.
    path : str, optional
        The path to the output file. The default is "./outputs2/".
    """
    print("Running all instances...")
    processes = []
    for instance_num in range(1,10):
        processes.append(Process(target=run_instance, args=(deepcopy(instance_num), time, path)))
        processes[-1].start()
    for instance_num, process in enumerate(processes):
        process.join()
        content = ""
        if os.path.exists(path):
            with open(f"{path}output_{instance_num+1}.txt", "r") as file:
                content = file.read()
        date = datetime.now().isoformat()
        if not os.path.exists(f"./outputs_instance{instance_num+1}/"):
            os.mkdir(f"outputs_instance{instance_num+1}")
        with open(f"./outputs_instance{instance_num+1}/log_{date}.txt", "w") as file:
            file.write(content)
    print("All instances done !")


def format_solution(layout: list, fitness:int):
    res = ""
    for i in range(len(layout)):
        res += str(layout[i]) + " "
    res += str(fitness)
    return res


#print(format_solution(np.array([19, 16, 20, 22,  5, 12, 11, 23, 24, 26, 13,  8,  4, 17, 10,  9,  2, 25, 14,  6]), 42))

def run_instance(instance_num, time=500, path="./outputs2/"):
    """
    Run an instance

    Parameters
    ----------
    instance_num : int
        The instance number.
    time : float, optional
        The time limit for the instance. The default is 500.
    path : str, optional
        The path to the output file. The default is "./outputs2/".
    """
    print(f"Running instance {instance_num}...")
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

    n=num_qubit
    m=backend.num_qubits

    # Number of times n fits in m
    nbr_layouts = m // n

    # Generate the layouts, we restrain each threads to a certain interval to limit the number of duplicates
    layouts = [np.random.choice(m - (n * i), n, replace=False) for i in range(nbr_layouts)]

    if not os.path.exists(path):
        os.mkdir(path)

    file = open(f"{path}output_{instance_num}.txt", "w")
    print(f"Beginning diversification for instance {instance_num}...")
    print("Starting threads...")
    file.write(f"Beginning diversification for instance {instance_num}...\n")
    file.write(f"Diversification starting ----------------\n")
    file.close()

    # Run the diversification in parallel
    process_count = 8
    bests = [[] for _ in range(process_count)]
    processes = []
    queue = Queue()
    for i in range(process_count):
        processes.append(NewProcess(queue, args=(layouts[i % nbr_layouts], 1000, 0.8, time / 2, bests[i], qr, qc, backend, 2, instance_num, m, i % nbr_layouts, process_count, path)))
        processes[-1].start()
    [process.join() for process in processes]

    # Merge the results of the threads
    bests = []
    for i in range(process_count):
        bests += queue.get()

    file = open(f"{path}output_{instance_num}.txt", "a")

    print(f"Diversification done for instance {instance_num} !")
    file.write(f"Diversification done for instance {instance_num} !\n")

    # Keep the best results
    bests.sort(key=lambda x: x[1])
    best_layouts = bests[:process_count]
    print(f"Best layouts : {best_layouts}")

    file.write(f"\n\nStarting threads for intensification of 5 bests...\n")
    print("Starting threads...")
    file.write(f"\nBests : {best_layouts}\n")
    file.write(f"Starting threads for intensification of bests...\n")
    file.write(f"Intensification starting ----------------\n")
    file.close()

    # Run the intensification in parallel
    bests_intensification = [[] for _ in range(min(process_count, len(best_layouts)))]
    processes = []
    queue = Queue()
    for i in range(min(process_count, len(best_layouts))):
        processes.append(NewProcess(queue, args=(best_layouts[i][0], 1, 0.9, time / 2, bests_intensification[i], qr, qc, backend, 1, instance_num, m, 0, process_count, path)))
        processes[-1].start()
    [process.join() for process in processes]

    file = open(f"{path}output_{instance_num}.txt", "a")
    file.write(f"Intensification done for instance {instance_num} !\n")

    # Merge the results of the threads
    i_results = []
    for i in range(min(process_count, len(best_layouts))):
        i_results += queue.get()

    i_results.sort(key=lambda x: x[1])

    print(f"Intensification done for instance {instance_num} !")
    file.write(f"Intensification done for instance {instance_num} !\n")
    file.write(f"\nBest layouts found after intensification :\n")
    file.write(f"{i_results[0][0]}\n")
    file.write(f"n={n}, m={m} et fitness_test={i_results[0][1]}. Instance {instance_num} !\n----------------\n")

    file.write(f"\n\nBest layout found :\n")
    file.write(f"{i_results[0][0]}\n")
    file.write(f"n={n}, m={m} et fitness_test={i_results[0][1]}. Instance {instance_num} !\n----------------\n")

    with open(f"5_Instance_{instance_num}.txt", 'w') as file1:
        file1.write(format_solution(i_results[0][0], i_results[0][1]))

    print(f"Best layout found : {i_results[0]}")

    file.close()

    return i_results[0][1] # Return the best fitness found




best = np.inf

from datetime import datetime
date = datetime.now().isoformat()

try:
    # Run all the instances in parallel for 25 minutes (we keep 20 seconds off for the small instructions not counted in the time limit)
    run_all_instances(1480, "./outputs_25_2min/") 
except KeyboardInterrupt:
    print("KeyboardInterrupt !")
    # Kill all the processes
    for process in active_children():
        process.terminate()


###### A faire : un algo d'optimisation qui minimise la fonction fitness,
###### fonction qui accepte en entrée :
###### une liste de n parmi m (n<=m) entiers deux à deux distincts
###### N'oubliez pas d'écrire la solution dans [GROUPE]_instance_[instance_num].txt
