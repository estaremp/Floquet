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

from numpy import linalg as LA

#---------------------FUNCTIONS--------------------------#

def chop(A, eps = 0.1):
    B = np.copy(A)
    B[np.abs(A) < eps] = 0
    return B

def evaluate_H(t,args):
    H1 = args[0]
    H2 = args[1]
    H3 = args[2]
    T = args[3]
    range = int(t/T)

    H = bool(((range-1) * T )<=t<=(range + T/3)) * H1 \
        + bool((range + T/3)<t<=(range + (2*T)/3)) * H2 \
        + bool((range + (2*T)/3)<t<=(range + T)) * H3

    return H

#Hamiltonian terms time-dependence
def coef_H1(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = int(t/T)
    return bool(((range-1) * T )<=t<=(range + T/3))

def coef_H2(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = int(t/T)
    return bool((range + T/3)<t<=(range + (2*T)/3))

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

N = 10
T = 1.0
g = (3.0 * pi) / (2.0)
eps = 0.0
alpha = 1.5
J0 = 0.108
W = (3.0 * pi)
t0 = 0.0
numframes = 2

#Time for the dynamics
times = np.linspace(0.0,T,10.0)
t0_list = np.linspace(0.0,T,numframes)

#Boolean options (more time)
animate = False

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
#or any other expectation value
fid = zeros(len(times))
for n, t in enumerate(times):
     psi_t = floquet_wavefunction_t(floquet_states, quasi_energies, floquet_coeff, t, H, T, args)
     fid[n] = fidelity(Psi0, psi_t)

fig, ax = plt.subplots(figsize=(10,6))

ax.plot(times, fid)

ax.set_xlabel(r'Time')
ax.set_ylabel(r'Fidelity against Psi0')
ax.set_title(r'Stroboscopic dynamics of the time crystal');
plt.show()

#Plot full dynamics
#dynamics.full_dynamics(N,H,Psi0,sx_exp_list[0],times)

#Plot full entanglement graph
#G = entanglement.entangled_graph(N,H,Psi0,10.0,10.0)

#-----------------------------------------------------------#
#------------------EVOLUTION WITH T0------------------------#
#-----------------------------------------------------------#

fig, ax = plt.subplots(figsize=(10,6))

eff_hami = []
maps = []
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
    if animate:
        map = ax.imshow(np.real(chop(eff_hami[k])),vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest', animated=True)
        plt.colorbar()
        maps.append([map])


if animate:
    ani = animation.ArtistAnimation(fig, maps, interval=1, blit=True, repeat_delay=1000)
    #Animation has to be run from the .html file
    ani.save('dynamic_map.html')

else:
    #Only show effective hamiltonian from [0,T]
    map = ax.imshow(np.real(chop(eff_hami[0])),vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest', animated=True)
    fig.colorbar(map,ax=ax)
    plt.show()

#-----------------------------------------------------------#
#------------------------NETWORK----------------------------#
#-----------------------------------------------------------#

# fig, ax = plt.subplots(figsize=(10,10))
# graphs = []
# for k,t0 in enumerate(t0_list):
#
#     #Create node label and their weight
#     node_list = []
#     for i in range(2**N):
#         node_list.append([i,eff_hami[k][i,i]])
#
#     #Create edge list and their weight
#     edges_list = []
#     for i in range(2**N):
#         for j in range(i+1,2**N):
#             edges_list.append([(i,j), eff_hami[k][i,j]])
#
#     #Create graph
#     G = nx.Graph()
#
#     G.add_edges_from([edges_list[i][0] for i in range(len(edges_list))])
#
#     #Specify weight for nodes and edges
#     vals = [round(np.abs(node_list[n][1]),1) for n in G.nodes()]
#     weights = [round(np.abs(edges_list[n][1]),3) for n in range(len(G.edges()))]
#
#     #Drwa graph
#     nodes = nx.draw_networkx_nodes(G, vmin=-1., vmax=1., cmap=plt.get_cmap('BuGn'), node_color=vals, width = weights,pos=nx.circular_layout(G))
#     edges = nx.draw_networkx_edges(G, vmin=-1., vmax=1., cmap=plt.get_cmap('BuGn'), node_color=vals, width = weights,pos=nx.circular_layout(G))
#
#     #Append group of graphs
#     graphs.append([nodes,edges,])
#
# #Animate graph evolution
# ani = animation.ArtistAnimation(fig, graphs, interval=1, blit=True, repeat_delay=1000)
# #animation has to be run from the .html file
# ani.save('dynamic_graph.html')

#-----------------------------------------------------------#
#---------------ENTANGLEMENT EVOLUTION----------------------#
#-----------------------------------------------------------#

fig, ax = plt.subplots(figsize=(10,10))

if animate:

    ent_graph = []
    for k,t0 in enumerate(t0_list):

        #Put Hamiltonians together
        args = {'t0': t0, 'T': T}
        H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

        #Find the floquet eigenstates and quasienergies
        floquet_states,quasi_energies = floquet_modes(H,T,args,True,None)

        eff = floquet_states[N]
        ent = np.zeros(shape=(N,N))
        for i in range(N):
            red_rho = 0
            for j in range(i+1,N):
                red_rho = eff.ptrace([i,j])
                ent[i,j] = concurrence(red_rho)

        #Create node label and their weight
        node_list = []
        for i in range(N):
            node_list.append([i,ent[i,i]])

        #Create edge list and their weight
        edges_list = []
        for i in range(N):
            for j in range(i+1,N):
                edges_list.append([(i,j), ent[i,j]])

        #Create graph
        G = nx.Graph()

        G.add_edges_from([edges_list[i][0] for i in range(len(edges_list))])

        #Specify weight for nodes and edges
        vals = [round(np.abs(node_list[n][1]),1) for n in G.nodes()]
        weights = [round(np.abs(edges_list[n][1]),3) for n in range(len(G.edges()))]

        #Drwa graph
        nodes = nx.draw_networkx_nodes(G, vmin=-1., vmax=1., cmap=plt.get_cmap('YlOrRd'), node_color=vals, width = weights,pos=nx.circular_layout(G))
        edges = nx.draw_networkx_edges(G, vmin=-1., vmax=1., cmap=plt.get_cmap('BuGn'), node_color=vals, width = weights,pos=nx.circular_layout(G))

        #Append group of graphs
        ent_graph.append([nodes,edges,])

    #Animate graph evolution
    ani = animation.ArtistAnimation(fig, ent_graph, interval=1, blit=True, repeat_delay=1000)
    #animation has to be run from the .html file
    ani.save('entangled_graph.html')

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

    #Create node label and their weight
    node_list = []
    for i in range(N):
        node_list.append([i,ent[i,i]])

    #Create edge list and their weight
    edges_list = []
    for i in range(N):
        for j in range(i+1,N):
            edges_list.append([(i,j), ent[i,j]])

    #Create graph
    G = nx.Graph()

    G.add_edges_from([edges_list[i][0] for i in range(len(edges_list))])

    #Specify weight for nodes and edges
    vals = [round(np.abs(node_list[n][1]),1) for n in G.nodes()]
    weights = [round(np.abs(edges_list[n][1]),3) for n in range(len(G.edges()))]

    #Drwa graph
    nodes = nx.draw_networkx_nodes(G, vmin=-1., vmax=1., cmap=plt.get_cmap('YlOrRd'), node_color=vals, width = weights,pos=nx.circular_layout(G))
    edges = nx.draw_networkx_edges(G, vmin=-1., vmax=1., cmap=plt.get_cmap('BuGn'), node_color=vals, width = weights,pos=nx.circular_layout(G))
    plt.savefig("state_%d.png" % k)