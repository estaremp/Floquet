from qutip import *
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def entangled_matrices(N,H,Psi0,t,args):

    #---------------------ENTANGLEMENT---------------------------#

    #Solve state
    result = mesolve(H, Psi0, t, [], [],args)

    entanglement = []

    #Calculate entanglement graph
    for s in result.states:
        ent = np.zeros(shape=(N,N))
        for i in range(N):
            red_rho = []
            for j in range(i+1,N):
                red_rho = s.ptrace([i,j])
                ent[i,j] = concurrence(red_rho)
        entanglement.append(ent)
    return entanglement