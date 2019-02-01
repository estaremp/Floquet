from qutip import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_weighted_graph(adj_matrix):

    N = adj_matrix.shape[0]

    #Create node label and their weight
    node_list = []
    for i in range(N):
        node_list.append([i,adj_matrix[i,i]])

    #Create edge list and their weight
    edges_list = []
    for i in range(N):
        for j in range(i+1,N):
            edges_list.append([(i,j), adj_matrix[i,j]])

    #Create graph
    G = nx.Graph()

    G.add_edges_from([edges_list[i][0] for i in range(len(edges_list))])

    #Specify weight for nodes and edges
    vals = [round(np.abs(node_list[n][1]),1) for n in G.nodes()]
    weights = [round(np.abs(edges_list[n][1]),3) for n in range(len(G.edges()))]

    #Draw graph
    nodes = nx.draw_networkx_nodes(G, vmin=-1., vmax=1., cmap=plt.get_cmap('BuGn'), node_color=vals, width = weights,pos=nx.circular_layout(G))
    edges = nx.draw_networkx_edges(G, vmin=-1., vmax=1., cmap=plt.get_cmap('BuGn'), node_color=vals, width = weights,pos=nx.circular_layout(G))

    return G, nodes, edges