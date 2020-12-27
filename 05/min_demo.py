"""
Finds the minimum of (Cx1 - x1)**2 + (Cx2 - x2)**2.
"""

from scipy.optimize import fmin_l_bfgs_b
import random
import numpy

def value(x, Cx1, Cx2):
    x1,x2 = x
    sc = (Cx1 - x1)**2 + (Cx2 - x2)**2 
    return sc

def grad(x, Cx1, Cx2):
    """ Return derivates for all dimensions. """
    x1,x2 = x
    return numpy.array([ -2*Cx1 + 2*x1, -2*Cx2 + 2*x2 ])

print(fmin_l_bfgs_b(value, #function to optimize
    x0=(0, 0), #staring solution
    args=(3.2453, 2.4), #arguments to all value and grad functions
                        #here the constants defining the function shape.
    fprime=grad)[0]) #function returning gradient(s)

