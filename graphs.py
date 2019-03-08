from qutip import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

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

    G.add_edges_from([edges_list[i][0] for i in range(len(edges_list)) if (edges_list[i][1])>0.001])

    #Specify weight for nodes and edges
    vals = [round(np.abs(node_list[n][1]),1) for n in G.nodes()]
    weights = [round((edges_list[n][1]),1) for n in range(len(edges_list)) if (edges_list[i][1])>0.001]

    pos = nx.circular_layout(G)

    #define basis
    states={}
    i=0
    for state in state_number_enumerate([3,3,3]):
        states[i]=str(state)
        i=i+1

    #Draw graph
    nodes = nx.draw_networkx_nodes(G, vmin=-1., vmax=1., cmap=plt.get_cmap('YlOrBr'), node_color=vals, pos=pos)
    edges = nx.draw_networkx_edges(G, width = weights, pos=pos)
    nx.draw_networkx_labels(G,pos,states,font_size=16)


    return G, nodes, edges