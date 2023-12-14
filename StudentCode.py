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

'''
Génération d'une population de façon aléatoire.
@param size : taille de la population de base
@param n : instance
@return une matrice contenant les individus de la population généré
'''
def firstPopu(size,n):
    layout=list(range(n))
    popu=[]
    for i in range(size):
        l=deepcopy(layout)
        random.shuffle(l)
        popu.append(l)
    
    return popu
    
'''
Mutation sur les individus.
@param popu : la population sur qui appliquer la mutation
@param lim : la probabilité de mutation
@error lim : ce paramètre doit être strictement plus petit que 1
'''
def mutageneParty(popu,lim):
    if lim/100 >= 1:
        raise ValueError("You cannot do this with lim="+lim+". It must be less than 1")
    for s in popu:
        if random.random()<=lim/100:
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

def afterParty(popu,lim):
    if lim/100 >= 1:
        raise ValueError("You cannot do this with lim="+lim+". It must be less than 1")
    for s in popu:
        if random.random()<=lim/100:
            p=random.randint(0,len(s)-1)
            q=random.randint(0,len(s)-1)
            if p!=q:
                s[p],s[q]=s[q],s[p]


'''
Croisement de deux parents donnant deux enfants
@param parent1 : un individus d'une population
@param parent2 : un individus d'une population
@param b : la probabilité de croisement
'''
def partial_match_crossover(parent1, parent2,b):
    if b/100<random.random():
        return parent1,parent2
    else:
        length = len(parent1)
        
        # Choisissez deux points de coupure aléatoires
        cut_point1, cut_point2 = sorted(random.sample(range(length), 2))
    
        # Initialisez les enfants en copiant les segments de parents entre les points de coupure
        child1 = parent1[cut_point1:cut_point2]
        child2 = parent2[cut_point1:cut_point2]
    
        # Complétez les enfants en préservant l'ordre des éléments non utilisés
        unused1 = [gene for gene in parent2 if gene not in child1]
        unused2 = [gene for gene in parent1 if gene not in child2]
    
        child1 += unused1
        child2 += unused2
    
        return child1, child2

'''
Sélection d'un couple d'individus d'une population. On a pas de doublons.
@param popu : une population
@param d : proportion de la séléction, d*2 est la taille de la séléction
@return view : une sélection dans la population
@error d : il doit être plus grand ou égale à 1
'''
def whoIsHorny(popu,d):
    if d<1:
        raise ValueError("the parametre \"d\" must be greater than 1")
    l=max(len(popu)//d,2)
    view=[]
    while l>0:
        i=random.randint(1,len(popu)-1)
        if i in view or i+1 in view:
            l-=1
        else:
            view.append(popu[i-1])
            view.append(popu[i])
            l-=1
    
    return view


'''
Engendre une nouvelle population après les possibles mutations et croisements.
@param popu : une population sur lequel on se base
@param a : probabilité de croisement
@param b : probabilité de mutation
@param d : proportion de la séléction
@return choosed : la nouvelle population
'''
def giveNewPopu(popu,a,b,d):

    print("-"*25+"On fait des gosses ici"+"-"*25)

    choosed=whoIsHorny(popu,d)
    i=0
    while i<len(choosed):
        k=partial_match_crossover(choosed[i],choosed[i+1],a)
        if k is None:
            break
        choosed[i]=k[0]
        choosed[i+1]=k[1]
        i+=2
    #mutageneParty(choosed,b)
    afterParty(choosed,b)
    return choosed

'''
Affiche les individus avec leur coût et donne le meilleur. On donne le coût du pire individus.
@param popu : la population sur qui, on demande les informations
@retrun lim : le plus grand coût de la population
'''
def allCost(popu,give):
    currMin=0
    cost=2002
    lim=-273
    i=0
    b=0
    w=0

    for s in popu:
        score=fitness(s)
        print(str(s)+"--->"+str(score)+"\n")
        i+=1
        if score<cost:
            cost=score
            currMin=s
            b=i
        elif score>lim:
            lim=score
            w=i-1
    
    if give:
        print("Le meilleur c'est le "+str(b)+"e avec un coût de "+str(cost)+".")
    
    return lim,popu[b]

    

'''
On applique la P-metaheuristique évolutionnaire. Elle comporte une séléction, un croisement et une mutation.
La séléction est aléatoire mais elle prend deux individus successifs.
Le croisement prend une partie de l'un et le transfert dans l'autre et inversement. La taille de la partie croisée est aléatoire.
La mutation consiste à inverser une partie de l'individus.
La nouvelle population est constitué des meilleurs individus à détail près.

@author Razanajao Aina
@param best : la population de base, la première génération
@param a : probabilité de croisement en pourcentage
@param b : probabilité de mutation en pourcentage
@param c : la réduction des probabilités en porcentage
@param d : proportion de la séléction
@param lim : limite de coût présumé pour la génération suivante
'''
def lawOfLife(best,a,b,c,d,lim=1918,last=False):
    allCost(best,True)
    kids=giveNewPopu(best,a/100,b/100,d)
    newLim,better=allCost(kids,True)

    if newLim<lim:
        lim=newLim

    for p in best:
        if fitness(p)>=lim:
            best.remove(p)
    
    newPopu=best+kids
    popuClear=[]
    for s in newPopu:
        if s not in popuClear:
            popuClear.append(s)

    if (not last) and (len(popuClear)<5 or a<10 or b < 10):
        print("-"*25+"OK le meilleur va faire le multi-clonage !"+"-"*25)
        kageJibunNoJutsu(better,78,69,6,6)
    elif last:
        print("-"*25+"On va s'arrêter là je crois"+"-"*25)
        allCost(popuClear,True)
    else:
        print("-"*25+"Ah shit, here we go again"+"-"*25)
        lawOfLife(popuClear,a-c,b-c,c,lim)

def kageJibunNoJutsu(best,a,b,c,d):
    clone=[]
    for i in range(10):
        clone.append(best)

    clone=whoIsHorny(clone,d)
    i=0
    while i<len(clone):
        k=partial_match_crossover(clone[i],clone[i+1],a)
        if k is None:
            break
        clone[i]=k[0]
        clone[i+1]=k[1]
        i+=2
    #mutageneParty(choosed,b)
    afterParty(clone,b)
    allCost(clone,True)



lawOfLife(firstPopu(16,n),80,58,6,4)



'''
Mon algorithme évolutif à une séléction et une mutation à tendance non diversifiant. De plus, la probabilité de mutation et de croisement baisse de plus en plus.
Cela implique une convergence vers une stabilité au niveau des individus. Concernent le croisement, la taille du croisement est aléatoire. Je choisis de laisser
la liberté au code de se diversifier ou non. Je tiens à faire disparaître les doutes, le code ne génère pas de doublons dans un individus. En addition à cela, la
nouvelle population devrait être les meilleurs entre l'ancienne population et de la nouvelle. Néanmoins, le code refait un fitness() sur l'ancienne population.
L'affichage de la population finale implique un appelle à la fonction allCost() refaisant un fitness sur la dernière population. Due à ma problématique et à mes
faibles connaissances concernant les ordinateurs quantiques, je dois admettre que les coûts d'un individu calculés à des moments différents peuvent être différent.
Cela ne garantit pas que mon code donne le meilleur coût obtenu mais il assure que l'individus est le meilleur.
'''
