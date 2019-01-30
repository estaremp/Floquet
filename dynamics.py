from qutip import *
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def full_dynamics(N,H,Psi0,expect,t):

    #------------------------DYNAMICS------------------------#

    #Solve master equation
    result = mesolve(H, Psi0, t, [], expect)

    #---------------------PLOT---------------------------#

    #Plot magnetization
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(t, np.real(result.expect[0]))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'\langle\sigma_x\rangle')
    ax.set_title(r'Full dynamics of the time crystal');
    plt.show()