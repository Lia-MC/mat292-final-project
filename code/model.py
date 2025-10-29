import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Example: dy/dt = -k * y
def model(y, t, k):
    dydt = -k * y
    return dydt

