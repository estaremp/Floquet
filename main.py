#import qutip
from qutip import *

import numpy as np
import matplotlib.pyplot as plt

#This creates a new quantum Object
print(Qobj()) #this is a 1x1 matrix with a single zero entry

#Create a QObj with a user defined data
print(Qobj([[1],[2],[3],[4],[5]])
