#######################################################################
#                                                                     #
#  Copyright 2014 Cristian C Lalescu                                  #
#                                                                     #
#  This file is part of pyNT.                                         #
#                                                                     #
#  pyNT is free software: you can redistribute it and/or modify       #
#  it under the terms of the GNU General Public License as published  #
#  by the Free Software Foundation, either version 3 of the License,  #
#  or (at your option) any later version.                             #
#                                                                     #
#  pyNT is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of     #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      #
#  GNU General Public License for more details.                       #
#                                                                     #
#  You should have received a copy of the GNU General Public License  #
#  along with pyNT.  If not, see <http://www.gnu.org/licenses/>       #
#                                                                     #
#######################################################################

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from wiener import Wiener, get_t1ma_nm1

class SDE(object):
    def __init__(self):
        return None
    def EM(self, W, X0):
        X = np.zeros(tuple([W.W.shape[0]] +
                           list(X0.shape) +
                           list(W.solution_shape)),
                     X0.dtype)
        X[0, :] = X0.reshape(X0.shape + (1,)*len(W.solution_shape))
        for t in range(1, W.nsteps + 1):
            X[t] = (X[t-1]
                  + self.drift(X[t-1])*W.dt)
            b = self.vol(X[t-1])
            for j in range(W.noise_dimension):
                X[t] += b[:, j]*(W.W[t, j] - W.W[t-1, j])
        return X
    def get_evdt_vs_M(
            self,
            fig_name = 'tst',
            ntraj = 32,
            X0 = None,
            h0 = 2.**(-3),
            exp_range = range(8)):
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_axes([.1, .1, .8, .8])
        bla = Wiener(
                nsteps = 2**8,
                dt = h0 / (2**8),
                noise_dimension = self.get_noise_dimension(),
                solution_shape = [200, ntraj])
        bla.initialize()
        if type(X0) == type(None):
            X0 = np.zeros(self.get_system_dimension(), dtype = np.float)
        full_wiener_paths = [bla.coarsen(2**n)
                        for n in exp_range]
        for M in [10, 20, 30, 40, 60, 100, 200]:
            wiener_paths = []
            for w in full_wiener_paths:
                new_w = w.coarsen(n = 1)
                new_w.W = w.W[:, :, :M]
                new_w.solution_shape = [M, ntraj]
                new_w.shape = [w.noise_dimension] + new_w.solution_shape
                wiener_paths.append(new_w)
            dtlist = [wiener_paths[p].dt for p in range(len(wiener_paths))]
            xnumeric = [self.EM(wiener_paths[p], X0) for p in range(len(wiener_paths))]
            err = [np.abs(xnumeric[p+1][-1] - xnumeric[p][-1]) / np.abs(xnumeric[p][-1])
                   for p in range(len(xnumeric)-1)]
            erri = [np.average(errij, axis = 1) for errij in err]
            averr  = [np.average(err[p]) for p in range(len(err))]
            sigma2 = [np.sum((averr[p] - erri[p])**2) / (M - 1)
                      for p in range(len(err))]
            deltae = [get_t1ma_nm1(0.99, M - 1)*(sigma2[p] / M)**.5
                      for p in range(len(err))]
            ax.errorbar(
                    dtlist[:-1],
                    averr,
                    yerr = deltae,
                    marker = '.',
                    label = 'M = {0}'.format(M))
        ax.plot(dtlist[:-1], dtlist[:-1], label = '$\\Delta t$')
        ax.plot(dtlist[:-1], np.array(dtlist[:-1])**.5, label = '$\\Delta t^{1/2}$')
        ax.set_xlabel('$\\Delta t$')
        ax.set_ylabel('$\\epsilon$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc = 'best')
        fig.savefig(fig_name + '.pdf', format = 'pdf')
        return dtlist
    def get_noise_dimension():
        return None
    def get_system_dimension():
        return None

class dwell(SDE):
    def __init__(
            self,
            c = 0.,
            n = 1.):
        self.c = c
        self.noise = n
        return None
    def get_noise_dimension(
            self):
        return 1
    def get_system_dimension(
            self):
        return 2
    def drift(
            self,
            x):
        return np.array([x[1],
                        -(4*x[0]**3 + 3*self.c*x[0]**2 - 2*x[0] + x[1])])
    def vol(
            self,
            x):
        tmp = np.ones(x.shape, x.dtype)
        return [np.array([np.zeros(x[0].shape, x.dtype)*0,
                          np.ones (x[1].shape, x.dtype)*self.noise])]

class linSDE(SDE):
    def __init__(
            self,
            c0,
            c1):
        self.c0 = c0
        self.c1 = c1
        return None
    def drift(self, x):
        return self.c0*x[0]
    def vol(self, x):
        return [self.c1*x[0]]
    def exact(self, W, X0):
        time = W.get_time()
        return X0*np.exp(
                    (self.c0 - .5*self.c1**2)*time[:, np.newaxis, np.newaxis]
                    + self.c1*W.W)
    def get_noise_dimension(self):
        return 1
    def get_system_dimension(self):
        return 1

class addSDE(SDE):
    def __init__(
            self,
            c0,
            c1):
        self.c0 = c0
        self.c1 = c1
        return None
    def drift(self, x):
        return self.c0*x[0]
    def vol(self, x):
        return [self.c1*np.ones(x[0].shape, x.dtype)]
    def get_noise_dimension(self):
        return 1
    def get_system_dimension(self):
        return 1

class spSDE(SDE):
    def __init__(
            self,
            x = None,
            a = None,
            b = None):
        self.x = x
        self.a = a
        self.b = b
        self.drift_func = [sp.utilities.lambdify(tuple(self.x), sp.sympify(ak), np)
                           for ak in self.a]
        self.vol_func = [[sp.utilities.lambdify(tuple(self.x), sp.sympify(bkj), np)
                          for bkj in bk]
                         for bk in b]
        return None
    def drift(self, x):
        return np.array([self.drift_func[k](*tuple(x))
                         for k in range(len(self.x))])
    def vol(self, x):
        result = np.zeros((x.shape[0],) + (len(self.b[0]),) + x.shape[1:], dtype = x.dtype)
        for k in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[k, j] = self.vol_func[k][j](*tuple(x))
        return result
    def get_noise_dimension(self):
        return len(self.b[0])
    def get_system_dimension(self):
        return len(self.x)

