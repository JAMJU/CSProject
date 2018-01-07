import numpy as np
import math
import random as rd


class S2GD(object):
    def __init__(self, max_number_stoch, stepsize, lower_bound, functions, derivates, data_dim, x0):
        """
        This class computes a semi stochastic gradient descent
        :param max_number_stoch: number maximal of stochastic step per epoch
        :param stepsize: step size of the gradient descent
        :param lower_bound: lower bound on mu, the strong convexity constant of f
        :param functions: all the fi in a list
        :param derivates: all the the fi' in a list
        :param data_dim: dimension of the data
        """
        self.n = len(functions)
        self.m = max_number_stoch
        self.h = stepsize
        self.nu = lower_bound
        self.beta = sum([(1. - self.nu*self.h)**(float(self.m - t)) for t in range(1, self.m + 1)])
        self.proba = [(1 - self.nu*self.h)**(float(self.m - t)) / self.beta for t in range(1, self.m + 1)]
        self.intervals = np.cumsum(self.proba)

        self.f = functions
        self.f_der = derivates
        self.dimension = data_dim
        self.x = np.copy(x0)

        self.follow_loss = list()


    def draw_t(self):
        """ return randomly a integer value between 1 and m with proba defined in self.proba for each value"""
        u = rd.random()
        for t in range(self.m):
            if t == 0:
                if 0. <= u and u <= self.intervals[t]:
                    return t + 1
            else:
                if self.intervals[t - 1] <= u and u <= self.intervals[t]:
                    return t + 1

    def calculate_loss(self):
        return sum([self.f[i](self.x) for i in range(self.n)])

    def algorithm(self, horizon):
        for j in range(horizon):
            if j%10==0:
                print "############# step ",j
            self.g = 1./float(self.n)* np.sum(np.asarray([self.f_der[i](self.x) for i in range(self.n)]), axis = 0)

            self.y = np.copy(self.x)
            T = self.draw_t()
            for t in range(T):
                i = np.random.choice(self.n)
                self.y = self.y -self.h*(self.g + self.f_der[i](self.y) - self.f_der[i](self.x))

            self.x = np.copy(self.y)
            self.follow_loss.append(self.calculate_loss())
        return self.x

