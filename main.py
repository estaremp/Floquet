#import qutip
from qutip import *
import math

import numpy as np

pi = math.pi

#parameters
N = 2

g = (3.0 * pi) / 2.0
eps = 0.0
alpha = 1.51
J0 = 0.11
W = 3.0 * pi

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

print(H1)

#Define H2
for i in range(N):
    for j in range(i+1,N):
        H2 += ( J0 / (abs(i-j)**alpha) ) * sx_list[i] * sx_list[j]

print(H2)

#Define H3
for i in range(N):
    H3 += W * D[i] * sx_list[i]

print(H3)