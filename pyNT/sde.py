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
from .wiener import Wiener, get_t1ma_nm1

def Lj(x, a, b, f, j):
    return sum(b[k][j]*sp.diff(f, x[k])
               for k in range(len(x)))

def Stratonovich_drift(x, a, b):
    return [(a[i] - sum(Lj(x, a, b, b[i][j], j)
                        for j in range(len(b[i])))/2)
            for i in range(len(x))]

def SL0(x, a, b, f):
    return sum(a[k]*sp.diff(f, x[k])
                for k in range(len(x)))

class base_SDE(object):
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
    def Milstein(self, W, X0):
        d = self.get_system_dimension()
        m = self.get_noise_dimension()
        X = np.zeros(tuple([W.W.shape[0]] +
                           list(X0.shape) +
                           list(W.solution_shape)),
                     X0.dtype)
        X[0, :] = X0.reshape(X0.shape + (1,)*len(W.solution_shape))
        ljb = []
        for i in range(d):
            ljb.append([])
            for j1 in range(m):
                ljb[i].append([sp.utilities.lambdify(
                                   tuple(self.x),
                                   Lj(self.x, self.a, self.b, self.b[i][j2], j1),
                                   np)
                               for j2 in range(m)])
        for t in range(1, W.nsteps+1):
            Jj = (W.W[t] - W.W[t-1])
            Jj0, J0j, Jjj, Ijj = W.get_jj(Jj)
            b = self.vol(X[t-1])
            X[t] = (X[t-1] +
                    W.Delta*self.drift(X[t-1]) +
                    np.array([sum(Jj[j]*b[i, j]
                                  for j in range(m))
                              for i in range(d)]) +
                    np.array([sum(sum(Ijj[j1,j2]*ljb[i][j1][j2](*tuple(X[t-1]))
                                      for j1 in range(m))
                                  for j2 in range(m))
                              for i in range(d)]))
        return X
    def explicit_1p0(self, W, X0):
        d = self.get_system_dimension()
        m = self.get_noise_dimension()
        X = np.zeros(tuple([W.W.shape[0]] +
                           list(X0.shape) +
                           list(W.solution_shape)),
                     X0.dtype)
        X[0, :] = X0.reshape(X0.shape + (1,)*len(W.solution_shape))
        for t in range(1, W.nsteps+1):
            Jj = W.W[t] - W.W[t-1]
            Jj0, J0j, Jjj, Ijj = W.get_jj(Jj)
            aa = self.drift(X[t-1])
            bb = self.vol(  X[t-1])
            X[t] = X[t-1] + W.Delta*aa
            y = [X[t] + bb[:, j]*W.sqrtD
                 for j in range(m)]
            X[t] += (sum(Jj[j]*bb[:, j] for j in range(m))
                         + np.array([sum(sum(Ijj[j1,j2]*(self.vol_func[i][j2](*tuple(y[j1])) - bb[i, j2])
                                             for j1 in range(m))
                                         for j2 in range(m))
                                     for i in range(d)])/W.sqrtD)
        return X
    def explicit_1p5_additive(self, W, X0):
        d = self.get_system_dimension()
        m = self.get_noise_dimension()
        X = np.zeros(tuple([W.W.shape[0]] +
                           list(X0.shape) +
                           list(W.solution_shape)),
                     X0.dtype)
        X[0, :] = X0.reshape(X0.shape + (1,)*len(W.solution_shape))
        for t in range(1, W.nsteps+1):
            Jj = W.W[t] - W.W[t-1]
            Jj0, J0j, Jjj, Ijj = W.get_jj(Jj)
            aa = self.drift(X[t-1])
            bb = self.vol(  X[t-1])
            yp = [X[t-1] + W.Delta*aa/m + W.sqrtD*bb[:, j]
                  for j in range(m)]
            ym = [X[t-1] + W.Delta*aa/m - W.sqrtD*bb[:, j]
                  for j in range(m)]
            ap = np.array([[self.drift_func[i](*tuple(yp[j])) for j in range(m)]
                           for i in range(d)])
            am = np.array([[self.drift_func[i](*tuple(ym[j])) for j in range(m)]
                           for i in range(d)])
            Jj0 /= 2*W.sqrtD
            X[t] += (X[t-1] +
                     sum(bb[:, j]*Jj[j] +
                         (ap[:, j] - am[:, j])*Jj0[j] +
                         (ap[:, j] - 2*(m-2.)*aa/m + am[:, j])*W.Delta/4
                         for j in range(m)))
        return X
    def Taylor_2p0_additive(self, W, X0):
        d = self.get_system_dimension()
        m = self.get_noise_dimension()
        X = np.zeros(tuple([W.W.shape[0]] +
                           list(X0.shape) +
                           list(W.solution_shape)),
                     X0.dtype)
        X[0, :] = X0.reshape(X0.shape + (1,)*len(W.solution_shape))
        l0a = []
        lja = []
        ljja = []
        driftS = Stratonovich_drift(self.x, self.a, self.b)
        for i in range(d):
            l0a.append(sp.utilities.lambdify(
                           tuple(self.x),
                           SL0(self.x, driftS, self.b, driftS[i]),
                           np))
            lja.append([])
            ljja.append([])
            for j1 in range(m):
                lja[i].append(sp.utilities.lambdify(
                                  tuple(self.x),
                                  Lj(self.x, driftS, self.b, driftS[i], j1),
                                  np))
                ljja[i].append([])
                for j2 in range(m):
                    ljja[i][j1].append(sp.utilities.lambdify(
                                           tuple(self.x),
                                           Lj( self.x,
                                               driftS,
                                               self.b,
                                               Lj(self.x, driftS, self.b, driftS[i], j2),
                                               j1),
                                           np))
        for t in range(1, W.nsteps+1):
            Jj = (W.W[t] - W.W[t-1])
            Jj0, J0j, Jjj, Jjj0, Ijj = W.get_jjj(Jj)
            aa = self.drift(X[t-1])
            bb = self.vol(  X[t-1])
            X[t] = (X[t-1]
                    + aa*W.Delta
                    + sum(Jj[j]*bb[:, j]
                          for j in range(m))
                    + np.array([l0a[i](*tuple(X[t-1]))
                                for i in range(d)])*(W.Delta**2)/2
                    + np.array([sum(Jj0[j]*lja[i][j](*tuple(X[t-1]))
                                    for j in range(m))
                                for i in range(d)])
                    + np.array([sum(sum(Jjj0[j1,j2]*ljja[i][j1][j2](*tuple(X[t-1]))
                                        for j1 in range(m))
                                    for j2 in range(m))
                                for i in range(d)]))
        return X
    def get_evdt_vs_M(
            self,
            fig_name = None,
            ntraj = 32,
            X0 = None,
            h0 = 2.**(-3),
            exp_range = range(8),
            solver = ['Milstein', 'explicit_1p0']):
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
        wiener_paths = [bla.coarsen(2**n)
                        for n in exp_range]
        dtlist = [wiener_paths[p].dt for p in range(len(wiener_paths))]
        random_state = np.random.get_state()
        xnumeric0 = [getattr(self, solver[0])(wiener_paths[p], X0) for p in range(len(wiener_paths))]
        np.random.set_state(random_state)
        xnumeric1 = [getattr(self, solver[1])(wiener_paths[p], X0) for p in range(len(wiener_paths))]
        err = [(np.average(np.abs(xnumeric0[p][-1] - xnumeric1[p][-1]), axis = 0) /
                np.average(np.abs(xnumeric0[p][-1]), axis = 0))
               for p in range(len(xnumeric0))]
        for M in [10, 20, 30, 40, 60, 100, 200]:
            erri = [np.average(errij[:M], axis = 1) for errij in err]
            averr  = [np.average(erri[p]) for p in range(len(erri))]
            sigma2 = [np.sum((averr[p] - erri[p])**2) / (M - 1)
                      for p in range(len(erri))]
            deltae = [get_t1ma_nm1(0.99, M - 1)*(sigma2[p] / M)**.5
                      for p in range(len(erri))]
            ax.errorbar(
                    dtlist,
                    averr,
                    yerr = deltae,
                    marker = '.',
                    label = 'M = {0}'.format(M))
        ax.plot(dtlist, dtlist, label = '$\\Delta t$')
        ax.plot(dtlist, np.array(dtlist)**.5, label = '$\\Delta t^{1/2}$')
        ax.plot(dtlist, np.array(dtlist)**1.5, label = '$\\Delta t^{3/2}$')
        ax.plot(dtlist, np.array(dtlist)**2, label = '$\\Delta t^2$')
        ax.set_xlabel('$\\Delta t$')
        ax.set_ylabel('$\\epsilon$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Distance between {0} and {1} vs time step'.format(solver[0], solver[1]))
        ax.legend(loc = 'best')
        if not type(fig_name) == type(None):
            fig.savefig(fig_name + '.pdf', format = 'pdf')
        return dtlist, averr, deltae
    def get_noise_dimension():
        return None
    def get_system_dimension():
        return None

class SDE(base_SDE):
    def __init__(
            self,
            x = None,
            a = None,
            b = None):
        self.x = x
        self.a = [sp.sympify(ak) for ak in a]
        self.b = [[sp.sympify(bkj) for bkj in bk] for bk in b]
        self.drift_func = [sp.utilities.lambdify(tuple(self.x), ak,  np)
                           for ak in self.a]
        self.vol_func   =[[sp.utilities.lambdify(tuple(self.x), bkj, np)
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

