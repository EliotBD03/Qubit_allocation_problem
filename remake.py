import multiprocessing as mtp
import time
import numpy as np


from qiskit import QuantumCircuit
from copy import deepcopy
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

instance_num=8   #### Entre 1 et 9 inclue

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



START_TIME = time.time()

class NewProcessDiversification(mtp.Process):
    def __init__(self, target : callable, queue : mtp.Queue, args=()):
        super().__init__()
        self.daemon = True
        self.target = target
        self.queue = queue
        self.args = args
    
    def run(self):
        try:
            res = self.target(*self.args)
            self.queue.put(res)
        except KeyboardInterrupt:
            print("KeyboardInterrupt ! ")

class NewProcessIntensification(NewProcessDiversification):

    def __init__(self, target : callable, queue : mtp.Queue, layout1, layout2, args=()):
        super().__init__(target, queue, args)
        self.layout1 = layout1
        self.layout2 = layout2
    
    def run(self):
        try:
            res = self.target(cross_solutions(self.layout1, self.layout2),*self.args)
            self.queue.put(res)
        except KeyboardInterrupt:
            print("KeyboardInterrupt ! ")

def cross_solutions(layout1, layout2):
    """
    Will cross two solutions to get 2 children
    The child with the best cost will be returned
    """
    random_index = np.random.RandomState().choice(layout1.size, layout1.size//2, replace=False)
    complement_index = [i for i in range(layout1.size) if i not in random_index]
    
    child1 = np.array([None] * layout1.size)
    child1[random_index] = layout1[random_index]
    
    child2 = np.array([None] * layout1.size)
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

    if fitness(child1) < fitness(child2):
        return child1
    return child2


def generate_new_layout_random(n, m):
    """
    Generate a new layout by generating a new random layout
    """
    new_layout = np.random.choice(m, n, replace=False)
    return new_layout

def simulated_annealing(initial_layout, temp=1, min_temp=0.1, alpha=0.99, time_limit=10, n=n, m=m, qc=qc, qr=qr, backend=backend):
    start_time = time.time()

    best_layout_encountered = initial_layout
    best_fitness_encountered = fitness(best_layout_encountered, qr, qc, backend)
    #print(f"current layout -> {best_layout_encountered}, {best_fitness_encountered}")
    while (time.time() - start_time) < time_limit and temp > min_temp:
        opponent = generate_new_layout_random(n, m)
        opponent_fitness = fitness(opponent, qr, qc, backend)
        if opponent_fitness < best_fitness_encountered or np.random.uniform(0,1,1) < np.exp((best_fitness_encountered - opponent_fitness)/temp):
            best_layout_encountered, best_fitness_encountered = opponent, opponent_fitness
            #print(f"better layout found ! -> {best_layout_encountered}, {best_fitness_encountered}")
        temp *= alpha
    
    return best_layout_encountered, best_fitness_encountered


def choose_pairs(layouts):
    parents = [None] * len(layouts)
    blacklist = [[]] * len(layouts)
    for i in range(len(layouts)):
        random_index = np.random.randint(0,len(layouts) - 1)
        while i != random_index and i in blacklist[random_index] : random_index = np.random.randint(0, len(layouts) - 1)
        parents[i] = (layouts[i][0], layouts[random_index][0])
        blacklist.append(i)
    return parents


def make_instance(num_instance=instance_num, nb_of_processes=mtp.cpu_count(), max_time=120):

    backend_name,circuit_type,num_qubit=instance_selection(num_instance)
    backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)
    n=num_qubit
    m=backend.num_qubits


    print(n, m)

    print(f"Running instance {num_instance} ({(time.time() - START_TIME) / 60} min)")

    processes = [None] * nb_of_processes
    queue = mtp.Queue()

    layouts = [np.random.RandomState().choice(m, n, replace=False) for _ in range(nb_of_processes)]

    mid_time = max_time // 2
    start_time = time.time()

    while time.time() - start_time < mid_time:
        for i in range(nb_of_processes):
            processes[i] = NewProcessDiversification(target=simulated_annealing, queue=queue, args=(layouts[i],10,0.1, 0.90, mid_time % nb_of_processes, n, m, qc, qr, backend))
            processes[i].start()
        [process.join() for process in processes]

    results_obtained = []
    while not queue.empty():
        results_obtained.append(queue.get())

    pairs = choose_pairs(sorted(results_obtained, key=lambda t : t[1])[:nb_of_processes])
    queue = mtp.Queue()
    for i in range(nb_of_processes):
        processes[i] = NewProcessIntensification(target=simulated_annealing, queue=queue, layout1= pairs[i][0], layout2=pairs[i][1], args=(1, 0.1, 0.99, mid_time % nb_of_processes, n, m, qc, qr, backend))
        processes[i].start()
    
    [process.join() for process in processes]
    results_obtained = sorted([queue.get() for _ in range(nb_of_processes)], key=lambda t : t[1])
    print(f"Best result found for the instance {num_instance} : {results_obtained[0][0]} with a cost of {results_obtained[0][1]} ({(time.time() - START_TIME) / 60} min)")
    return (results_obtained[0][0], results_obtained[0][1])

def make_instances(instances : list[int]):
    processes = [None] * len(instances)
    for i in range(len(processes)):
        processes[i] = mtp.Process(target=make_instance, args=(deepcopy(instances[i]),))
        processes[i].start()
    processes[i].join()

if __name__ == "__main__":
    make_instance()
