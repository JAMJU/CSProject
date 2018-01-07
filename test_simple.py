import matplotlib.pyplot as plt
import math
import numpy as np
from S2GD import S2GD
import random as rd
############## Test with simple function

def quadratic(x):
    return x[0]**2

def otherquadra(x):
    return 2.*(x[1] - 1.)**2

def deriv1(x):
    x0 = np.asscalar(x[0])
    x1 = np.asscalar(x[1])
    print x0,x1
    return np.asarray([[2*x0], [0.]])

def deriv2(x):

    x0 = np.asscalar(x[0])
    x1 = np.asscalar(x[1])
    print x0, x1
    return np.asarray([[0.], [4.*(x1 - 1.)]])

# Creation of the functions and gradient we are going to use:
print "start creation of function"
f = list()
f_der = list()

fun = lambda x: quadratic(x)
f.append(fun)
fun_der = lambda x: deriv1(x)
f_der.append(fun_der)

fun = lambda x: otherquadra(x)
f.append(fun)
fun_der = lambda x: deriv2(x)
f_der.append(fun_der)

algo1 = S2GD( max_number_stoch = 50, stepsize = 0.1, lower_bound = 1., functions = f, derivates = f_der,
              data_dim = 2, x0 = np.asarray([[rd.random()], [rd.random()]],  dtype = np.float))
final_x = algo1.algorithm(100)

# We plot the evolution of the loss function
plt.plot(algo1.follow_loss)
plt.show()

print algo1.x