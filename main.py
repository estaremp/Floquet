#import qutip
from qutip import *
from scipy import *
from scipy.linalg import logm, expm
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#import our subroutines
import entanglement
import dynamics
import graphs


#---------------------FUNCTIONS--------------------------#

def chop(A, eps = 0.1):
    B = np.copy(A)
    B[np.abs(A) < eps] = 0
    return B

#time dependence for H1
def coef_H1(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = int(t/T)
    return bool(((range-1) * T )<=t<=(range + T/3))

#time dependence for H2
def coef_H2(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = int(t/T)
    return bool((range + T/3)<t<=(range + (2*T)/3))

#time dependence for H3
def coef_H3(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = int(t/T)
    return bool((range + (2*T)/3)<t<=(range + T))

pi = np.pi

#----------------------------------------------------------#
#------------------------PARAMETERS------------------------#
#----------------------------------------------------------#

#Hamiltonian parametrization
N = 4                      #size
T = 1.0                    #Period
g = (3.0 * pi) / (2.0)     #g H1
eps = 0.0                  #error in H1
alpha = 1.5                #distribution coupling
J0 = 0.108                 #maximum coupling
W = (3.0 * pi)             #strength of random disorder

#Dynamical conditions
t0 = 0.0                   #initial time
tF = 10.0                  #final time
tStep = 100                #time step
nT = 10                    #num of stroboscopic steps
numframes = 10             #num frames for animations

#Time for the dynamics
t_list = np.linspace(0.0,tF,tStep)
t0_list = np.linspace(0.0,T,numframes)

#Boolean options (more time)
evolve_t0 = False
all_eigenstates = True

#Entanglement calculation
x = N                        #entanglement graph of the x eigenstate

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

#Define H1 (time-independent componets)
for i in range(N):
    H1 += g * (1-eps) * sy_list[i]

#Define H2 (time-independent componets)
for i in range(N):
    for j in range(i+1,N):
        H2 += ( J0 / (abs(i-j)**alpha) ) * sx_list[i] * sx_list[j]

#Define H3 (time-independent componets)
for i in range(N):
    H3 += W * D[i] * sx_list[i]

#-----------------------------------------------------------#
#------------------------MEASURES---------------------------#
#-----------------------------------------------------------#

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

#Put Hamiltonians together
args = {'t0': t0, 'T': T}
H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

#-----------------------------------------------------------#
#------------------------FLOQUET----------------------------#
#-----------------------------------------------------------#

#Find Floquet operator
Floquet_op = propagator(H, T, [], args)

#Find the floquet eigenstates and quasienergies
floquet_states,quasi_energies = floquet_modes(H,T,args,True,None)

#Decompose initial state into them
floquet_coeff = floquet_state_decomposition(floquet_states,quasi_energies,Psi0)


#-----------------------------------------------------------#
#--------------STROBOSCOPIC DYNAMICS------------------------#
#-----------------------------------------------------------#

#NOTE: WE CONSIDER T0 = 0.0 HERE

#Time evolve stroboscopically and get fidelity against initial state
dynamics.stroboscopic_dynamics(floquet_states,quasi_energies,floquet_coeff,H,T,nT,Psi0,args)

#Plot full dynamics for t0 = 0
dynamics.full_dynamics(H,Psi0,sx_exp_list[0],t_list,args)

#Plot full entanglement dynamics
entanglement.entangled_matrices(N,H,Psi0,t_list,args)

#TODO - obtain stroboscopic entangled graph evolution
#TODO - obtain full entangled graph evolution

#-----------------------------------------------------------#
#------------------EVOLUTION Heff WITH T0-------------------#
#-----------------------------------------------------------#

fig1, ax1 = plt.subplots(figsize=(10,6))
fig2, ax2 = plt.subplots(figsize=(10,10))

if evolve_t0:
    eff_hami = []
    maps = []
    graph = []
    for k,t0 in enumerate(t0_list):

        #Put Hamiltonians together
        args = {'t0': t0, 'T': T}
        H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

        #Obtain Floquet operator
        Floquet_op = propagator(H, T, [], args)

        #Effective hamiltonian
        eff_h = ((-1/(1.j*T))*logm(Floquet_op.full()))

        #Obtain list of effective Hamiltonians
        eff_hami.append(eff_h)

        #Animate matrixplot and graph of effective hamiltonian with t0
        map = ax1.imshow(np.real(chop(eff_hami[k])),vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest', animated=True)
        maps.append([map])

        #Generate graphs
        G, nodes, edges = graphs.generate_weighted_graph(np.real(chop(eff_hami[k])))

        #Append group of graphs
        graph.append([nodes,edges,])

    #Animate effective hamiltonian evolution
    fig1.colorbar(map,ax=ax1)
    ani_heff = animation.ArtistAnimation(fig1, maps, interval=1, blit=True, repeat_delay=1000)
    #Animation has to be run from the .html file
    ani_heff.save('dynamic_map.html')

    #Animate graph evolution
    ani_graph = animation.ArtistAnimation(fig2, graph, interval=1, blit=True, repeat_delay=1000)
    #animation has to be run from the .html file
    ani_graph.save('dynamic_graph.html')


else:
    #Only show effective hamiltonian from [0,T]

    #Effective hamiltonian
    eff_h = ((-1/(1.j*T))*logm(Floquet_op.full()))

    #Graph
    G, nodes, edges = graphs.generate_weighted_graph(np.real(chop(eff_h)))
    fig2.savefig('effective_graphT0.png')

    map = ax1.imshow(np.real(chop(eff_h)),vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest', animated=True)
    fig1.colorbar(map,ax=ax1)
    fig1.savefig('effective_hami_mapT0.png')

#-----------------------------------------------------------#
#---------------ENTANGLEMENT EVOLUTION----------------------#
#-----------------------------------------------------------#

#entanglement graph for x Floquet eigenstate
fig, ax = plt.subplots(figsize=(10,10))

if evolve_t0:
    ent_graph = []
    for k,t0 in enumerate(t0_list):

        #Put Hamiltonians together
        args = {'t0': t0, 'T': T}
        H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

        #Find the floquet eigenstates and quasienergies at each t0
        floquet_states,quasi_energies = floquet_modes(H,T,args,True,None)

        eff = floquet_states[x]
        ent = np.zeros(shape=(N,N))
        for i in range(N):
            red_rho = 0
            for j in range(i+1,N):
                red_rho = eff.ptrace([i,j])
                ent[i,j] = concurrence(red_rho)

        #Generate graphs
        G, nodes, edges = graphs.generate_weighted_graph(ent)

        #Append group of graphs
        ent_graph.append([nodes,edges,])

    #Animate graph evolution
    ani = animation.ArtistAnimation(fig, ent_graph, interval=1, blit=True, repeat_delay=1000)
    #animation has to be run from the .html file
    ani.save('entangled_graph_vectX.html')
else:
    #Only show effective hamiltonian from [0,T]
    eff = floquet_states[0]
    ent = np.zeros(shape=(N,N))
    for i in range(N):
        red_rho = 0
        for j in range(i+1,N):
            red_rho = eff.ptrace([i,j])
            ent[i,j] = concurrence(red_rho)

    #Generate graphs
    G, nodes, edges = graphs.generate_weighted_graph(ent)
    plt.savefig('entangled_graph_vectX-T0.png')

#get entanglement graph for all eigenstates t0
if all_eigenstates:
    for k in range(2**N):

        #Put Hamiltonians together
        args = {'t0': t0, 'T': T}
        H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

        #Find the floquet eigenstates and quasienergies
        floquet_states,quasi_energies = floquet_modes(H,T,args,True,None)

        eff = floquet_states[k]
        ent = np.zeros(shape=(N,N))
        for i in range(N):
            red_rho = 0
            for j in range(i+1,N):
                red_rho = eff.ptrace([i,j])
                ent[i,j] = concurrence(red_rho)

        #Generate graphs
        G, nodes, edges = graphs.generate_weighted_graph(ent)

        plt.savefig("entangled grap_vect%d-T0.png" % k)