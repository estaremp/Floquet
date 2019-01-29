#import qutip
from qutip import *
import math

import numpy as np

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

pi = math.pi

#parameters
N = 2
g = (3.0 * pi) / 2.0
eps = 0.0
alpha = 1.51
J0 = 0.11
W = 3.0 * pi
T = 1

si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sx_list = []
sy_list = []
sz_list = []

D = []

for i in range(N):
    D.append(np.random.uniform(0.0, 1.0))

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

Hargs = (H1,H2,H3,T)

t = 1.4

energies = evaluate_H(t,Hargs).eigenenergies()

