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
from copy import deepcopy
import random

##-------------------------------------------------------
##     Definition de la fonction objetif à minimiser
##-------------------------------------------------------
def fitness(layout):
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

###### A faire : un algo d'optimisation qui minimise la fonction fitness,
###### fonction qui accepte en entrée :
###### une liste de n parmi m (n<=m) entiers deux à deux distincts
###### N'oubliez pas d'écrire la solution dans [GROUPE]_instance_[instance_num].txt

###### /!\ Attention /!\
###### Il est possible que la ligne 40 : qasmfile=f".\Instances\{l.rstrip()}.qasm"
###### crée un problème à l'execution si le chemin n'est pas le bon,
###### en particulier sous Linux. Essayer de la remplacer par
###### qasmfile=f"./Instances/{l.rstrip()}.qasm". Cela devrait résoudre le problème.

def firstPopu(size,n):
    layout=list(range(n))
    popu=[]
    for i in range(size):
        l=deepcopy(layout)
        random.shuffle(l)
        popu.append(l)
    
    allCost(popu)
    return popu

def mutageneParty(popu,lim):
    if lim >= 1:
        raise ValueError("You cannot do this with lim="+lim+". It must be less than 1")
    for s in popu:
        if random.random()<=lim:
            p=random.randint(0,len(s)-1)
            q=random.randint(0,len(s)-1)
            if p!=q:
                if p<q: 
                    tmp=s[p:q]
                    tmp.reverse()
                    s[p:q]=tmp
                else:
                    tmp=s[q:p]
                    tmp.reverse()
                    s[q:p]=tmp


def before(l):
    return random.randint(0,len(l)//2)

def after(l):
    return random.randint(len(l)//2+1,len(l)-1)

def nsfw(parents,lim):
    if lim >= 1:
        raise ValueError("You cannot do this with lim="+lim+". It must be less than 1")
    if random.random()<=lim:
        k1=deepcopy(parents[0])
        k2=deepcopy(parents[1])

        x=before(k1)
        y=after(k1)

        part1=k1[x:y]
        part2=k2[x:y]

        k1[x:y]=part2
        k2[x:y]=part1
        
        return k1,k2

def whoIsHorny(popu):
    l=len(popu)//5
    view=[]
    while l!=0:
        i=random.randint(1,len(popu)-1)
        if i in view or i+1 in view:
            l-=1
        else:
            view.append(popu[i-1])
            view.append(popu[i])
            l-=1
    
    return view

def giveNewPopu(popu,a,b):

    print("-"*20+"On fait des gosses ici"+"-"*20)

    choosed=whoIsHorny(popu)
    i=0
    while i<len(choosed):
        p=[choosed[i],choosed[i+1]]
        k=nsfw(p,a)
        if k is None:
            break
        choosed[i]=k[0]
        choosed[i+1]=k[1]
        i+=2
    
    mutageneParty(choosed,b)
    return choosed

def allCost(popu):
    currMin=0
    cost=2002
    i=0
    b=0
    for s in popu:
        i+=1
        score=fitness(s)
        print(str(s)+"-->"+str(score)+"\n")
        if score<cost:
            cost=score
            currMin=s
            b=i

    print("Le meilleur c'est le "+str(b)+"e avec un coût de "+str(cost)+".")
    

allCost(giveNewPopu(firstPopu(20,n),0.85,0.32))


def lawOfLife(first,a,b):
    allCost(giveNewPopu(first,a,b))

