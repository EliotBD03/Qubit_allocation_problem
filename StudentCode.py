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
    Generate a new layout by generating a new random layout
    """
    new_layout = np.random.choice(m - (len(layout) * cur_layout), len(layout), replace=False)
    return new_layout

def generate_new_layout_add(layout, m, cur_layout):
    """
    Generate a new layout by adding a new element to the layout
    """
    new_layout = layout.copy()
    new_element = np.random.randint(m - (len(layout) * cur_layout))
    new_layout = list(map(lambda x: (x + new_element) % m, new_layout))
    return new_layout


"""
Current best results :
[19, 9, 16, 10, 4, 1, 5, 14, 12, 7, 15, 3, 11, 2, 0, 18, 13, 8, 17, 6]
n=20, m=27 et fitness_test=63. Instance 1 ok !
"""
target = 22
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
        The type of simulated annealing to use. The default is 1 (swap two elements, intensification). Other values are 2 (generate a new random layout, diversification).
    """
    print(f"Starting simulated annealing... Type : {_type}. Instance {instance}")
    local_seed = np.random.RandomState()
    local_seed.seed()
    np.random.seed(local_seed.randint(0, 1000000000))
    layout1 = layout.copy()
    T_copy = deepcopy(T)
    fitness1 = fitness(layout1, qr=qr, qc=qc, backend=backend)
    best_fitness = fitness1
    best_layout = layout1
    bests_list = [(best_layout, best_fitness)]

    rand_values = np.random.rand(int(time_limit * 10)).tolist()  # Precompute random values

    print(f"Simulated annealing starting for instance {instance}...")

    start_time = time.time()

    # Run the algorithm for 'time_limit' seconds (of thread actual running time)
    while time.time() - start_time < time_limit and best_fitness > target:
        # Generate a new layout
        match _type:
            case 1:
                new_layout1 = generate_new_layout_swap_two(layout1)
            case 2:
                new_layout1 = generate_new_layout_random(layout1, m, cur_layout)
            case _:
                new_layout1 = generate_new_layout_add(layout1, m, cur_layout)

        new_fitness1 = fitness(new_layout1, qr=qr, qc=qc, backend=backend)

        if new_fitness1 < fitness1 or rand_values.pop() < np.exp((fitness1 - new_fitness1) / T_copy):
            layout1, fitness1 = new_layout1, new_fitness1
            bests_list.append((layout1, fitness1))
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

    # Keep the 5 bests and the 3 worsts
    bests_list.sort(key=lambda x: x[1])

    return bests_list[:nbr_process]
"""
def cross_solutions(layout1, layout2):
    
    Will cross two solutions to get 2 children
    The child with the best cost will be returned
    
    random_index = np.random.choice(n, n//2, replace=False)
    complement_index = [i for i in range(n) if i not in random_index]
    print(f"layout1 -> {layout1}, layout2 -> {layout2}")
    child1 = layout1[random_index]
    print(f"child1 -> {child1}")
    complement_part_child1 = layout2[complement_index]
    print(f"complement_part_child1 -> {complement_part_child1}")
    child2 = layout2[random_index]
    complement_part_child2 = layout1[complement_index]
    
    for index in complement_index:
        i = 0
        while i < n and complement_part_child1[i] in child1: i += 1
        if i == n:
            random_nb = np.random.randint(0, m - 1)
            while random_nb in child1 : np.random.randint(0, m - 1)
            child1[index] = random_nb
        else:
            child1[index] = complement_part_child1[i]

        i = 0
        while i < n and complement_part_child2[i] in child2: i += 1
        if i == n:
            random_nb = np.random.randint(0, m - 1)
            while random_nb in child2 : np.random.randint(0, m - 1)
            child2[index] = random_nb
        else:
            child2[index] = complement_part_child2[i]
    
    if fitness(child2) < fitness(child1):
        return child2
    return child1
"""      

def cross_solutions(layout1, layout2):
    """
    Will cross two solutions to get 2 children
    The child with the best cost will be returned
    """
    random_index = np.random.choice(n, n//2, replace=False)
    complement_index = [i for i in range(n) if i not in random_index]
    
    child1 = np.array([None] * n)
    child1[random_index] = layout1[random_index]
    
    child2 = np.array([None] * n)
    child2[random_index] = layout2[random_index]

    complement_part_child1 = layout2[complement_index]
    complement_part_child2 = layout1[complement_index]
    
    for index in complement_index:
        i = 0
        while i < complement_part_child1.size and complement_part_child1[i] in child1: i += 1 #check if a value from layout2 is available
        if i == complement_part_child1.size: #if it's not the case, we generate a random value
            random_nb = np.random.randint(0, m - 1) 
            while random_nb in child1 : random_nb = np.random.randint(0, m - 1)
            child1[index] = random_nb
        else:
            child1[index] = complement_part_child1[i] #Otherwise, we take it

        i = 0 #We do the same for the second child
        while i < complement_part_child2.size and complement_part_child2[i] in child2: i += 1
        if i == complement_part_child2.size:
            random_nb = np.random.randint(0, m - 1)
            while random_nb in child2 : random_nb = np.random.randint(0, m - 1)
            child2[index] = random_nb
        else:
            child2[index] = complement_part_child2[i]
    children = [child1, child2]

    return children[np.random.randint(0,1)]


def run_all_instances(time=100, path="./outputs2/"):
    print("Running all instances...")
    processes = []
    for instance_num in range(2):
        processes.append(Process(target=run_instance, args=(deepcopy(instance_num), time, path)))
        processes[-1].start()
    for instance_num, process in enumerate(processes):
        process.join()
        with open(f"{path}output_{instance_num+1}.txt", "r") as file:
            content = file.read()
        date = datetime.now().isoformat()
        with open(f"./outputs_instance{instance_num+1}/log_{date}.txt", "w") as file:
            file.write(content)
    print("All instances done !")

def run_instance(instance_num, time=500, path="./outputs2/"):
    print(f"Running instance {instance_num}...")
    backend_name,circuit_type,num_qubit=instance_selection(instance_num)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

    n=num_qubit
    m=backend.num_qubits

    # Number of times n fits in m
    nbr_layouts = m // n

    # Generate the layouts, we restrain each threads to a certain interval to limit the number of duplicates
    layouts = [np.random.choice(m - (n * i), n, replace=False) for i in range(nbr_layouts)]

    file = open(f"{path}output_{instance_num}.txt", "w")
    print(f"Beginning diversification for instance {instance_num}...")
    print("Starting threads...")
    file.write(f"Beginning diversification for instance {instance_num}...\n")
    file.write(f"Diversification starting ----------------\n")
    file.close()
    process_count = 1
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

    bests.sort(key=lambda x: x[1])

    best_layouts = bests[:process_count]
    print(f"Best layouts : {best_layouts}")


    crossed_best_layouts = [[] for _ in range(len(best_layouts))]
    blacklist = [[]] * len(crossed_best_layouts)
    for i in range(len(crossed_best_layouts)):
        random_index = np.random.randint(0,len(crossed_best_layouts) - 1)
        while i != random_index and i in blacklist[random_index] : random_index = np.random.randint(0, len(crossed_best_layouts) - 1)
        print(f"best_layouts[i] {best_layouts[i]}")
        crossed_best_layouts[i] = cross_solutions(best_layouts[i][0], best_layouts[random_index][0])
        blacklist.append(i)
    print(f"generated children from previous solutions : {crossed_best_layouts}")
    bests_intensification = [[] for _ in range(min(process_count, len(best_layouts)))]
    file.write(f"\n\nStarting threads for intensification of 5 bests...\n")
    print("Starting threads...")
    file.write(f"\nBests : {best_layouts}\n")
    file.write(f"Starting threads for intensification of bests...\n")
    file.write(f"Intensification starting ----------------\n")
    file.close()
    processes = []
    queue = Queue()
    for i in range(min(process_count, len(best_layouts))):
        processes.append(NewProcess(queue, args=(crossed_best_layouts[i], 1, 0.9, time / 2, bests_intensification[i], qr, qc, backend, 1, instance_num, m, 0, process_count, path)))
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

    print(f"Best layout found : {i_results[0]}")

    file.close()

    return i_results[0][1]


best = np.inf

from datetime import datetime
date = datetime.now().isoformat()


try:
    run_all_instances(120, "./outputs_2min/")
except KeyboardInterrupt:
    print("KeyboardInterrupt !")
    # Kill all the processes
    for process in active_children():
        process.terminate()
