from qutip import *
from scipy import *
import numpy as np
import matplotlib.pyplot as plt

def full_dynamics(H,Psi0,expect,t_list,args):

    #------------------------DYNAMICS------------------------#

    #Solve master equation
    result = mesolve(H, Psi0, t_list, [], expect, args)

    #---------------------PLOT---------------------------#

    #Plot magnetization
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(t_list, np.real(result.expect[0]))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Expectation value')
    ax.set_title(r'Full dynamics of the time crystal');
    plt.savefig('full_dynamics.png')

def stroboscopic_dynamics(f_s,f_e,f_c,H,T,nT,Psi0,args):

    #Time evolve stroboscopically and get fidelity against initial state
    #or any other expectation value
    fid = zeros(nT)
    for t in range(nT):
        psi_t = floquet_wavefunction_t(f_s, f_e, f_c, t, H, T, args)
        fid[t] = fidelity(Psi0, psi_t)

    #---------------------PLOT---------------------------#

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot([t for t in range(nT)], fid)

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Fidelity against Psi0')
    ax.set_title(r'Stroboscopic dynamics of the time crystal');
    plt.savefig('stroboscopic_dynamics.png')