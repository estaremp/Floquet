#import qutip
from qutip import *
from scipy import *
from scipy.linalg import logm, expm
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz
import matplotlib.animation as animation

#import our subroutines
import entanglement
import dynamics as dyn
import graphs
import spectrum as sp

from networkx.drawing.nx_agraph import graphviz_layout


#---------------------FUNCTIONS--------------------------#

def chop(A, tol = 0.01):
    B = np.copy(A)
    B[np.abs(A) < tol] = 0
    return B

#time dependence for H1
def coef_H1(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = t % (T)
    return bool((range <= (T)/2))

#time dependence for H2
def coef_H2(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = t % (T)
    return bool((((T)/2)<range<=(3*(T)/4)))

#time dependence for H3
def coef_H3(t,args):
    t0 = args['t0']
    T = args['T']
    t = t + t0
    range = t % (T)
    return bool((3*(T)/4)<range<=(T))

pi = np.pi

#----------------------------------------------------------#
#------------------------PARAMETERS------------------------#
#----------------------------------------------------------#

#Hamiltonian parametrization
N = 3                      #size
T = 1.0                    #Period
g = (3.0*pi)/2.0           #g H1
eps = 1.0                 #error in H1
alpha = 1.5                #distribution coupling
J0 = 0.108                  #maximum coupling
r0 = 0.8
W = (2*pi)                 #strength of random disorder

#Dynamical conditions
t0 = 0.                    #initial time
tF = 10.                    #final time
numStep = 1000             #time steps
nT = 10                     #num of stroboscopic steps
numframes = 100            #num frames for animations

#Time for the dynamics
t_list = np.linspace(0.0,tF,numStep)
t0_list = np.linspace(0.0,T,numframes)
epsilons = np.linspace(0.0,0.5,numframes)

#Boolean options (more time)
floquet = True
dynamics = True
evolve_t0 =  False
all_eigenstates = False
epsilon_dy = False
stats = False
statistics_dyn = False
ent = False

#Entanglement calculation
x = N                        #entanglement graph of the x eigenstate

#-----------------------------------------------------------#
#------------------------HAMILTONIAN------------------------#
#-----------------------------------------------------------#

print('*** COMPOSING YOUR HAMILTONIAN ***')

#Tensor pauli matrices
si = qeye(3)
sm = destroy(3)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

s_a = basis(3,0)
s_b = basis(3,1)
s_c = basis(3,2)

sbc = s_b*s_c.dag()
scb = s_c*s_b.dag()
sab = s_a*s_b.dag()
sba = s_b*s_a.dag()
sac = s_a*s_c.dag()
sca = s_c*s_a.dag()
saa = s_a*s_a.dag()
scc = s_c*s_c.dag()


sbc_list = []
sab_list = []
scc_list = []
saa_list = []
sbci_list = []
scbi_list = []
sabi_list = []
sbai_list = []

D_min = []
D_max = []

#Define random disorder
for i in range(N):
    D_min.append(np.random.uniform(0.0, W))
    D_max.append(np.random.uniform(0.0, W))

#Define tensor products
for i in range(N):
    op = []
    for j in range(N):
        op.append(si)

    op[i] = (scb + sbc)
    sbc_list.append(tensor(op))

    op[i] = (sab + sba)
    sab_list.append(tensor(op))

    op[i] = scc
    scc_list.append(tensor(op))

    op[i] = saa
    saa_list.append(tensor(op))

    op[i] = sbc
    sbci_list.append(tensor(op))

    op[i] = scb
    scbi_list.append(tensor(op))

    op[i] = sab
    sabi_list.append(tensor(op))

    op[i] = sba
    sbai_list.append(tensor(op))


#Initi hami components
H1 = 0
H2 = 0
H1_1 = 0
H1_2 = 0
H3 = 0

#Define H1 (time-independent componets)
for i in range(N):
    H3 += (2*(eps)*pi)*sbc_list[i]

#Define H2 (time-independent componets)
for i in range(N):
    H2 += (2*(eps)*pi)*sab_list[i]

#Define H3 (time-independent componets)
for i in range(N):
    H1_1 += D_min[i]*(scc_list[i]) + D_max[i]*(saa_list[i])

for i in range(N):
    for j in range(i+1,N):
        H1_2 += (  0.01 / (abs(i-j)**3)  ) * ((-((scbi_list[i] * sbci_list[j]) + (sabi_list[i] * sbai_list[j])+(sbci_list[i] * scbi_list[j]) + (sbai_list[i] * sabi_list[j]))/2.0) + ((scc_list[i]-saa_list[i])*(scc_list[j]-saa_list[j])))

H1 = H1_1 + H1_2

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

#Define population
sa_exp_list = []
for i in range(N):
    op = []
    for j in range(N):
        op.append(si)

    op[i] = s_a * s_a.dag()
    sa_exp_list.append(tensor(op))

sb_exp_list = []
for i in range(N):
    op = []
    for j in range(N):
        op.append(si)

    op[i] = s_b * s_b.dag()
    sb_exp_list.append(tensor(op))

sc_exp_list = []
for i in range(N):
    op = []
    for j in range(N):
        op.append(si)

    op[i] = s_c * s_c.dag()
    sc_exp_list.append(tensor(op))

#Define initial state Psi0
psi_list = []
for i in range(N):
    psi_list.append(s_b)


#define basis
s=[]
for state in state_number_enumerate([3,3,3]):
    s.append(state)

Psi0 = tensor(psi_list)

#Put Hamiltonians together
args = {'t0': t0, 'T': T}
H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]


#-----------------------------------------------------------#
#------------------------FLOQUET----------------------------#
#-----------------------------------------------------------#

print('*** APPLYING FLOQUET THEORY ***')

#Find Floquet operator
Floquet_op = propagator(H, T, [], args)

#Find the floquet eigenstates and quasienergies
floquet_states,quasi_energies = floquet_modes(H,T,args,True,None)

#Decompose initial state into them
floquet_coeff = floquet_state_decomposition(floquet_states,quasi_energies,Psi0)


#Plot spectrum
sp.plot_spectrum(quasi_energies)



#-----------------------------------------------------------#
#--------------STROBOSCOPIC DYNAMICS------------------------#

#-----------------------------------------------------------#

#NOTE: WE CONSIDER T0 = 0.0 HERE

print('*** OBTAINING DYNAMICS ***')

if dynamics:

    #Time evolve stroboscopically and get fidelity against initial state
    dyn.stroboscopic_dynamics(floquet_states,quasi_energies,floquet_coeff,H,T,nT,sb_exp_list[0],args)

    exp = [sa_exp_list[0],sb_exp_list[0],sc_exp_list[0]]

    #Plot full dynamics for t0 = 0
    dyn.full_dynamics(H,Psi0,exp,t_list,args)

    #Plot full entanglement dynamics
    #entanglement.entangled_matrices(N,H,Psi0,t_list,args)

#TODO - obtain stroboscopic entangled graph evolution
#TODO - obtain full entangled graph evolution

#-----------------------------------------------------------#
#------------------EVOLUTION Heff WITH T0-------------------#
#-----------------------------------------------------------#

print('*** CALCULATING EFFECTIVE HAMILTONIAN ***')

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
        G, nodes, edges = graphs.generate_weighted_graph(np.abs(chop(eff_hami[k])))

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

    egv,en = np.linalg.eigh(Floquet_op.full())

    #for i,e in enumerate(en):
    #    for j,ele in enumerate(e):
    #        if np.abs(ele)>=0.01:
    #            print(s[i],s[j],np.abs(ele))

    #Graph
    G, nodes, edges = graphs.generate_weighted_graph(np.abs(chop(eff_h)))
    fig2.savefig('effective_graphT0.png')

    c = 0
    for i in sorted(nx.connected_components(G), key=len, reverse=True):
        if len(i)==3:
            c=c+1
    print(c)

    #for i in G.edges:
    #    if ((np.abs(chop(eff_h[i[0],i[1]])))>=0.001):
            #print(s[i[0]],s[i[1]],(np.abs(chop(eff_h[i[0],i[1]]))))



    map = ax1.imshow(np.real(chop(eff_h)),vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest', animated=True)
    fig1.colorbar(map,ax=ax1)
    fig1.savefig('effective_hami_mapT0.png')


#-----------------------------------------------------------#
#---------------GRAPH CHANGES WITH EPS----------------------#
#-----------------------------------------------------------#

fig3, ax3 = plt.subplots(figsize=(10,6))
fig4, ax4 = plt.subplots(figsize=(10,10))

if epsilon_dy:
    eff_hami = []
    maps = []
    graph = []
    H1 = 0
    for k,epsi in enumerate(epsilons):

        #Define H1 (time-independent componets)
        for i in range(N):
            H1 += g * (1-epsi) * sy_list[i]

        #Put Hamiltonians together
        args = {'t0': t0, 'T': T}
        H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

        #Obtain Floquet operator
        Floquet_op = propagator(H, T, [], args, options=Options(nsteps=10000))

        #Effective hamiltonian
        eff_h = ((-1/(1.j*T))*logm(Floquet_op.full()))

        #Obtain list of effective Hamiltonians
        eff_hami.append(eff_h)

        #Animate matrixplot and graph of effective hamiltonian with t0
        map = ax3.imshow(np.real(chop(eff_hami[k])),vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest', animated=True)
        maps.append([map])

        #Generate graphs
        G, nodes, edges = graphs.generate_weighted_graph(np.real(chop(eff_hami[k])))

        #Append group of graphs
        graph.append([nodes,edges,])

        H1 = 0

    #Animate effective hamiltonian evolution
    fig3.colorbar(map,ax=ax3)
    ani_eps_heff = animation.ArtistAnimation(fig3, maps, interval=1, blit=True, repeat_delay=1000)
    #Animation has to be run from the .html file
    ani_eps_heff.save('dynamic_eps_map.html')

    #Animate graph evolution
    ani_eps_graph = animation.ArtistAnimation(fig4, graph, interval=1, blit=True, repeat_delay=1000)
    #animation has to be run from the .html file
    ani_eps_graph.save('dynamic_eps_graph.html')


#-----------------------------------------------------------#
#---------------LEVELS SPACE STATISTICS---------------------#
#-----------------------------------------------------------#

if stats:

    def PGOE(i):
        return (27/4)*(i+i**2)/((1+i+i**2)**(5/2))

    def POISS(i):
        return (2/((1+i)**2))

    fig = plt.figure()

    stats = []
    for epsi in epsilons:

        H1 = 0

        #Define H1 (time-independent componets)
        for i in range(N):
            H1 += g * (1-epsi) * sy_list[i]

        #Put Hamiltonians together
        args = {'t0': t0, 'T': T}
        H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

        #Find Floquet operator
        Floquet_op = propagator(H, T, [], args)

        #Find the floquet eigenstates and quasienergies
        floquet_states,quasi_energies = floquet_modes(H,T,args,True,None)

        #Reorder sequence of quasienergies
        quasi_energies.sort()

        #Define NN spacings
        s = []
        for i in range(len(quasi_energies)-1):
            s.append(quasi_energies[i+1]-quasi_energies[i])

        #Ratios
        r = []
        for u in range(len(s)-1):
            r.append(min(s[u],s[u+1])/max(s[u],s[u+1]))

        stats.append(r)

    def update_hist(num,stats):
        plt.cla()
        theo_goe = []
        theo_poiss = []
        xs = np.linspace(0.0,1.0)
        epsi = epsilons[num]
        print(epsi)
        for i in xs:
            theo_goe.append(PGOE(i))
            theo_poiss.append(POISS(i))

        r = stats[num]
        plt.ylim(0, 5)
        plt.plot(xs,theo_goe,label="GOE")
        plt.plot(xs,theo_poiss,label="Poisson")
        plt.hist(r,bins='sturges',density=True,animated=True,)
        plt.title(r'$\varepsilon$=%0.2f' %epsi)
        plt.autoscale(False)


    theo_goe = []
    theo_poiss = []
    xs = np.linspace(0.0,1.0)
    for i in xs:
        theo_goe.append(PGOE(i))
        theo_poiss.append(POISS(i))

    r = stats[0]
    plt.plot(xs,theo_goe,label="GOE")
    plt.plot(xs,theo_poiss,label="Poisson")
    plt.hist(r,bins='sturges',density=True,animated=True,)
    plt.title(r'$\varepsilon$=%0.2f' %epsi)

    #Animate effective hamiltonian evolution
    #ax.legend()
    #lt.show()
    stats_eps = animation.FuncAnimation(fig, update_hist, numframes, fargs=(stats,) )
    #Animation has to be run from the .html file
    stats_eps.save('stats_eps_dyn.html')


#-----------------------------------------------------------#
#---------------ENTANGLEMENT EVOLUTION----------------------#
#-----------------------------------------------------------#

print('*** CALCULATING ENTANGLEMENT GRAPH ***')

if ent:

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
        eff = floquet_states[x]
        ent = np.zeros(shape=(N,N))
        for i in range(N):
            red_rho = []
            for j in range(i+1,N):
                red_rho = eff.ptrace([i,j])
                ent[i,j] = concurrence(red_rho)

        #Generate graphs
        G, nodes, edges = graphs.generate_weighted_graph(ent)
        plt.savefig('entangled_graph_vectX-T0.png')

    #get entanglement graph for all eigenstates t0
    if all_eigenstates:
        for k in range(2**N):

            fig, ax = plt.subplots(figsize=(10,10))

            ent = 0

            #Put Hamiltonians together
            args = {'t0': t0, 'T': T}
            H = [[H1,coef_H1],[H2,coef_H2],[H3,coef_H3]]

            #Find the floquet eigenstates and quasienergies
            floquet_states,quasi_energies = floquet_modes(H,T,args,True,None)

            eff = floquet_states[k]
            ent = np.zeros(shape=(N,N))
            for i in range(N):
                red_rho = []
                for j in range(i+1,N):
                    red_rho = eff.ptrace([i,j])
                    ent[i,j] = concurrence(red_rho)

            #Generate graphs
            G, nodes, edges = graphs.generate_weighted_graph(ent)

            plt.savefig("entangled grap_vect%d-T0.png" % k)
            plt.close(fig)