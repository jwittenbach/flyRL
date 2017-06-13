from circular import CircularFun, CircularSpline
from models import ValueModel
from viz import value_plot

import matplotlib.pyplot as plt
import numpy as np

# value function
v = CircularSpline(3, 5, w=[0, 0, 0, 1, 0])

# movement penalty function
p = CircularFun(lambda x, a: a*x**2, p_circ=0.5, args=[10])

# model
m = ValueModel(v, p, 0, 10, 20)

# simulation
x = m.simulate(2000)
value_plot(x, m)
