#import qutip
from qutip import *
from scipy import *
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#import our subroutines
import entanglement
import dynamics

#---------------------FUNCTIONS--------------------------#

# def evaluate_H(t,args):
#     H1 = args[0]
#     H2 = args[1]
#     H3 = args[2]
#     T = args[3]
#     range = int(t/T)
#
#     H = bool(((range-1) * T )<=t<=(range + T/3)) * H1 \
#         + bool((range + T/3)<t<=(range + (2*T)/3)) * H2 \
#         + bool((range + (2*T)/3)<t<=(range + T)) * H3
#
#     return H

def coef_H1(t,args):
    T = 1
    range = int(t/T)
    return bool(((range-1) * T )<=t<=(range + T/3))

def coef_H2(t,args):
    T = 1
    range = int(t/T)
    return bool((range + T/3)<t<=(range + (2*T)/3))

def coef_H3(t,args):
    T = 1
    range = int(t/T)
    return bool((range + (2*T)/3)<t<=(range + T))

pi = np.pi

#----------------------------------------------------------#
#------------------------PARAMETERS------------------------#
#----------------------------------------------------------#

N = 4
T = 1.0
g = (3.0 * pi) / (2.0)
eps = 0.03
alpha = 1.5
J0 = 0.108
W = (3.0 * pi)

#Time for the dynamics
times = np.linspace(0.0,10.0,10.0)

#-----------------------------------------------------------#
#------------------------HAMILTONIAN------------------------#
#-----------------------------------------------------------#

#Tensor pauli matrices
si = qeye(2)
sm = destroy(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sx_list = []
sy_list = []
sz_list = []

D = []

#Define random disorder
for i in range(N):
    D.append(np.random.uniform(0.0, 1.0))

#Define tensor products
for i in range(N):
    op = []
    for j in range(N):
        op.append(si)

    op[i] = sx
    sx_list.append(tensor(op))

    op[i] = sy
    sy_list.append(tensor(op))

    op[i] = sz
    sz_list.append(tensor(op))


#Initi hami components
H1 = 0
H2 = 0
H3 = 0

#Define H1
for i in range(N):
    H1 += g * (1-eps) * sy_list[i]

#Define H2
for i in range(N):
    for j in range(i+1,N):
        H2 += ( J0 / (abs(i-j)**alpha) ) * sx_list[i] * sx_list[j]

#Define H3
for i in range(N):
    H3 += W * D[i] * sx_list[i]

#Put Hamiltonians together
Hargs = (H1,H2,H3,T)
H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]


#Define magnetization tensor
sx_exp_list = []
for i in range(N):
    op = []
    for j in range(N):
        op.append(si)

    op[i] = sx
    sx_exp_list.append(tensor(op))

#Define occupation prob expectation value
sm_exp_list = []
for i in range(N):
    op = []
    for j in range(N):
        op.append(si)

    op[i] = sm.dag() * sm
    sm_exp_list.append(tensor(op))

#Define initial state Psi0
psi_list = []
for i in range(N):
    psi_list.append((1/np.sqrt(2)) * (basis(2,0)+basis(2,1)))

Psi0 = tensor(psi_list)

#-----------------------------------------------------------#
#------------------------FLOQUET----------------------------#
#-----------------------------------------------------------#

#find the floquet eigenstates and quasienergies
floquet_modes,quasi_energies = floquet_modes(H,T,Hargs)

#decompose initial state into them
floquet_coeff = floquet_state_decomposition(floquet_modes,quasi_energies,Psi0)

p_ex = zeros(len(times))
for n, t in enumerate(times):
    psi_t = floquet_wavefunction_t(floquet_modes, quasi_energies, floquet_coeff, t, H, T)
    p_ex[n]=fidelity(Psi0, psi_t)
    print(fidelity(Psi0, psi_t))

fig, ax = plt.subplots(figsize=(10,6))

ax.plot(times, np.real(p_ex))

ax.set_xlabel(r'Time')
ax.set_ylabel(r'Fidelity against Psi0')
ax.set_title(r'Stroboscopic dynamics of the time crystal');
plt.show()

#Plot dynamics
#dynamics.full_dynamics(N,H,Psi0,sx_exp_list[0],times)

#Plot entanglement graph
#G = entanglement.entangled_graph(N,H,Psi0,10.0,10.0)
