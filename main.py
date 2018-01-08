import matplotlib.pyplot as plt
import numpy as np
from S2GD import S2GD

# I got the MNIST data on https://pjreddie.com/projects/mnist-in-csv/

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

count = 0

def logistic_loss(lamb, label, data, x):
    global count
    count += 1
    if np.isnan(np.asscalar((x.T).dot(data))):
        raise Exception()
    z = logistic(data, x)
    if z < 0 or z > 1:
        print(z)

    return -(label*np.log(z) + (1.-label)*(np.log( 1. - z))) + lamb/2*np.asscalar((x.T).dot(x))

def logistic_loss_grad(lamb, label, data, x):
    global count
    count +=1
    x_dot_data = np.asscalar((x.T).dot(data))
    z = logistic(data, x)

    return (z - label)*data + lamb*x

def logistic(data,x):
    x_dot_data = np.asscalar((x.T).dot(data))
    return (1./(1.+ np.exp(-x_dot_data))
            if x_dot_data > 0.
            else np.exp(x_dot_data)/(1. +  np.exp(x_dot_data))
            )
############# PARAMETERS
# same than the article, but need to consider kappa to know nb of stochastic steps, and still pb of computation : do log sum !
lamb = 0.05
size_wanted = 1000

#########################

# get the data
print("get the data")
data, label_nb = returnDataAsArray("data/mnist_train.csv")
print(data.shape)
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
print("number of 5:",nb5)
print("number of 8:", nb8)


new_data = np.asarray(new_data, dtype = np.float)
print(new_data.shape)
new_label = np.asarray(new_label, dtype = np.float)

# Creation of the functions and gradient we are going to use:
print("start creation of function")
eps = 10**(-10)

def logistic_loss(lamb, label, data, x):
    z = logistic(data, x)

    mis_label = 0
    if label == 1.:
        if z < eps:
            mis_label = -np.log(eps)
        else:
            mis_label = -np.log(z)
    else:
        if z > 1 -  eps:
            mis_label = -np.log(eps)
        else:
            mis_label = -np.log(1.-z)

    return mis_label + lamb/2*np.asscalar((x.T).dot(x))

def logistic_loss_grad(lamb, label, data, x):
    z = logistic(data, x)

    return (z - label)*data + lamb*x

def logistic(data,x):
    x_dot_data = np.asscalar((x.T).dot(data))
    #x_dot_data = min(max(x_dot_data, 10**3), -10**3)
    return (1./(1.+ np.exp(-x_dot_data))
            if x_dot_data > 0.
            else np.exp(x_dot_data)/(1. + np.exp(x_dot_data))
            ) + eps

def partial_loss(lamb, label, data):
    def func(x):
        return logistic_loss(lamb, label, data, x)
    return func
def partial_loss_grad(lamb, label, data):
    def func(x):
        return logistic_loss_grad(lamb, label, data, x)
    return func

f = list()
f_der = list()
for i in range(new_label.shape[0]):
    f.append(partial_loss(lamb, new_label[i], np.copy(new_data[i].reshape(dimension, 1))))
    f_der.append(partial_loss_grad(lamb, new_label[i], np.copy(new_data[i].reshape(dimension, 1))))



x0 = np.asarray([[(1. - 2.*np.random.uniform())] for j in range(dimension)],  dtype = np.float) / 1000

# Creation of the algo S2GD
print("parameters nu = 0")
h = 10. ** (-6)
n_epochs = 11
m = 17100
algo1 = S2GD( max_number_stoch = m, stepsize = h, lower_bound = 0., functions = f, derivates = f_der,
              data_dim = dimension, x0=x0)

print("algo")
final_x = algo1.algorithm(n_epochs)


def partial(list_, i):
    return [elem[i] for elem in list_]

# We plot the evolution of the loss function
plt.semilogy(partial(algo1.follow_loss, 0)[0:-1], np.asarray(partial(algo1.follow_loss, 1)[0:-1]) - algo1.follow_loss[-1][1], 'ko-')
plt.show()

result_label_f = [logistic(new_data[i].reshape(dimension, 1), algo1.x) for i in range(new_data.shape[0])]
result_label = [float(logistic(new_data[i].reshape(dimension, 1), algo1.x) > 0.5) for i in range(new_data.shape[0])]
print(result_label_f[0:10])
print(result_label[0:10])
print(new_label[0:10])
accuracy = float(sum([int(r == orig) for r, orig in zip(result_label, new_label)]))/float(new_data.shape[0])
print("ACCURACY", accuracy)
print("COUNT", algo1.count_grad)
