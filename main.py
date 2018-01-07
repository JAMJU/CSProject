import matplotlib.pyplot as plt
import math
import numpy as np
from S2GD import S2GD
import random as rd

# I got the MNIST data on https://pjreddie.com/projects/mnist-in-csv/
eps = 0.000001 # to help calculation

def returnDataAsArray(namefile):
    data = list()
    label = list()
    with open(namefile, "rb") as f:
        list_lines = f.readlines()

        for line in list_lines:
            line = line.decode("utf_8")
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            line = line.split(',')
            label.append(int(line[0]))
            data.append([float(line[i]) for i in range(1,len(line))])
    f.close()
    dataMat = np.asarray(data)
    return dataMat, label

def logistic_loss(lamb, label, data, x):
    #print -np.asscalar((x.T).dot(data))
    if -np.asscalar((x.T).dot(data)) < 0.:
        return -(-float(label)*math.log((1.+ math.exp(np.asscalar(-(x.T).dot(data)))) + eps ) +
                 (1.-float(label))*(math.log( 1. - 1./(1.+ math.exp(np.asscalar(-(x.T).dot(data)))) + eps))) + \
               lamb/2.*np.asscalar((x.T).dot(x))
    else:
        #print "positive", np.asscalar((x.T).dot(data)), math.exp(np.asscalar((x.T).dot(data)))
        return -(float(label) * math.log(math.exp(np.asscalar((x.T).dot(data)))/(1. +  math.exp(np.asscalar((x.T).dot(data)))) + eps) +
                 (1. - float(label)) * (math.log(1. - math.exp(np.asscalar((x.T).dot(data)))/(1 + math.exp(np.asscalar((x.T).dot(data)))))) + eps) + \
               lamb/2. * np.asscalar((x.T).dot(x))

def logistic_loss_grad(lamb, label, data, x):
    if np.asscalar(-(x.T).dot(data)) < 0.:
        #print "neg", (1./(1. + math.exp(np.asscalar(-(x.T).dot(data)))) - float(label))*data + lamb*x
        return (1./(1. + math.exp(np.asscalar(-(x.T).dot(data)))) - float(label))*data + lamb*x
    else:
        #print "pos", (math.exp(np.asscalar((x.T).dot(data)))/(1. + math.exp(np.asscalar((x.T).dot(data)))) - float(label))*data + lamb*x
        return (math.exp(np.asscalar((x.T).dot(data)))/(1. + math.exp(np.asscalar((x.T).dot(data)))) - float(label))*data + lamb*x

def logistic(data,x):
    print np.asscalar(-(x.T).dot(data))
    if np.asscalar(-(x.T).dot(data)) <0.:
        return 1./(1. + math.exp(np.asscalar(-(x.T).dot(data))))
    else:
        return math.exp(np.asscalar((x.T).dot(data)))/ (1. + math.exp(np.asscalar((x.T).dot(data))))

############# PARAMETERS
# same than the article, but need to consider kappa to know nb of stochastic steps, and still pb of computation : do log sum !
lamb = 0.05
size_wanted = 1000

#########################

# get the data
print "get the data"
data, label_nb = returnDataAsArray("data/mnist_train.csv")
print data.shape
dimension = data.shape[1] + 1
# We take only the 5 and the 8
new_data = list()
new_label = list()
nb5 = 0
nb8 = 0
for i in range(len(label_nb)):
    if len(new_label) < size_wanted:
        dat = list(data[i])
        dat.append(1.)
        if label_nb[i] == 5: # the label is going to be 1
            new_label.append(1.)
            new_data.append(dat)
            nb5 += 1
        elif label_nb[i] == 8:
            new_label.append(0.)
            new_data.append(dat)
            nb8+=1
print "number of 5:",nb5
print "number of 8:", nb8


new_data = np.asarray(new_data, dtype = np.float)
print new_data.shape
new_label = np.asarray(new_label, dtype = np.float)

# Creation of the functions and gradient we are going to use:
print "start creation of function"
f = list()
f_der = list()
for i in range(new_label.shape[0]):
    fun = lambda x: logistic_loss(lamb, new_label[i], new_data[i].reshape(dimension, 1), x)
    f.append(fun)
    fun_der = lambda x: logistic_loss_grad(lamb, new_label[i], new_data[i].reshape(dimension, 1), x)
    f_der.append(fun_der)

# Creation of the algo S2GD
# change lower bound : maybe that's the pb !
algo1 = S2GD( max_number_stoch = 100, stepsize = 0.1, lower_bound = 0., functions = f, derivates = f_der,
              data_dim = dimension, x0 = np.asarray([[(1. - 2.*rd.random())] for j in range(dimension)],  dtype = np.float))
final_x = algo1.algorithm(10000)

# We plot the evolution of the loss function
plt.plot(algo1.follow_loss)
plt.show()

result_label_f = [logistic(new_data[i].reshape(dimension, 1), algo1.x) for i in range(new_data.shape[0])]
result_label = [float(logistic(new_data[i].reshape(dimension, 1), algo1.x) > 0.5) for i in range(new_data.shape[0])]
print result_label_f[0:10]
print result_label[0:10]
print new_label[0:10]
accuracy = float(sum([int(r == orig) for r, orig in zip(result_label, new_label)]))/float(new_data.shape[0])
print "ACCURACY", accuracy
print algo1.x.shape







