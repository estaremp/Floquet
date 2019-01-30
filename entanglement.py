from qutip import *
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def entangled_graph(N,H,Psi0,t):

    #---------------------ENTANGLEMENT---------------------------#

    #Solve state
    result = mesolve(H, Psi0, t, [], [])

    red_rho = 0
    entanglement = []

    #Calculate entanglement graph
    for s in result.states:
        ent = np.zeros(shape=(N,N))
        for i in range(N):
            red_rho = 0
            for j in range(i+1,N):
                red_rho = s.ptrace([i,j])
                ent[i,j] = concurrence(red_rho)
        entanglement.append(ent)

    #---------------------PLOT ENTANGLEMENT GRAPH---------------------------#

    G = nx.from_numpy_matrix(entanglement[5],parallel_edges=False)
    pos=nx.circular_layout(G)

    nx.draw_networkx_nodes(G,pos,node_color='green',node_size=500)

    all_weights = []
    for (node1, node2, data) in G.edges(data=True):
        all_weights.append(data['weight'])

    unique_weights = list(set(all_weights))

    for weight in unique_weights:
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        width = weight
        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width)

    plt.show()
    return G