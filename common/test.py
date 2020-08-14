import numpy as np
from common.util import *


a = np.arange(2 * 2).reshape(2, 2)
print(a)
t = np.mean(a, axis=0)
print(t)