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

class base_ODE(object):
    def __init__(self):
        return None
    def Euler(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            X[t] = X[t-1] + self.rhs(X[t-1])*h
        return X
    def Heun(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            k1 = self.rhs(X[t-1])
            k2 = self.rhs(X[t-1] + h*k1)
            X[t] = X[t-1] + h*(k1 + k2)/2
        return X
    def cRK(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            k1 = self.rhs(X[t-1])
            k2 = self.rhs(X[t-1] + h*k1/2)
            k3 = self.rhs(X[t-1] + h*k2/2)
            k4 = self.rhs(X[t-1] + h*k3)
            X[t] = X[t-1] + h*(k1 + 2*(k2+k3) + k4)/6
        return X
    def Taylor2(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        d = len(self.x)
        llx = [sum(self.lx[j]*self.lx[i].diff(self.x[j])
                   for j in range(d))
               for i in range(d)]
        acc = [sp.utilities.lambdify(tuple(self.x), llx[i], np)
               for i in range(d)]
        for t in range(1, nsteps+1):
            velnum = self.rhs(X[t-1])
            accnum = np.array([acc[i](*tuple(X[t-1]))
                               for i in range(d)])
            X[t] = (X[t-1]
                    + velnum*    h
                    + accnum*(.5*h**2))
        return X
    def Taylor4(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        d = len(self.x)
        # copy list explicitly
        diffx    = [[self.lx[i] for i in range(d)]]
        diffxnum = [[sp.utilities.lambdify(tuple(self.x), self.lx[i], np)
                     for i in range(d)]]
        for n in range(4):
            newdiff = [sum(self.lx[j]*diffx[-1][i].diff(self.x[j])
                           for j in range(d))
                       for i in range(d)]
            diffx.append(newdiff)
            diffxnum.append([sp.utilities.lambdify(tuple(self.x),diffx[-1][i], np)
                             for i in range(d)])
        for t in range(1, nsteps+1):
            terms = [np.array([diffxnum[i][j](*tuple(X[t-1]))
                               for j in range(d)])*(h**(i+1))/sp.factorial(i+1)
                     for i in range(4)]
            X[t] = X[t-1] + sum(terms)
        return X
    def get_evdt(
            self,
            h0 = 2.**(-3),
            nsteps = 1,
            X0 = None,
            exp_range = range(8),
            solver = ['Euler', 'Taylor2'],
            relative = True):
        x0 = [getattr(self, solver[0])(h0*2.**(-n), nsteps * 2**n, X0) for n in exp_range]
        x1 = [getattr(self, solver[1])(h0*2.**(-n), nsteps * 2**n, X0) for n in exp_range]
        dist = [np.average(np.abs(x0[p][-1] - x1[p][-1]), axis = 0)
                for p in range(len(x0))]
        if relative:
            print 'hello'
            for p in range(len(x0)):
                dist[p] /= np.average(np.abs(x0[p][-1]), axis = 0)
        err_vs_dt = np.zeros((len(x0), 4), dtype = X0.dtype)
        err_vs_dt[:, 0] = np.array([h0*2.**(-n) for n in exp_range])
        err_vs_dt[:, 2] = np.average(dist, axis = tuple(range(1, len(X0.shape))))
        err_vs_dt[:, 1] = err_vs_dt[:, 2] - np.min(dist, axis = tuple(range(1, len(X0.shape))))
        err_vs_dt[:, 3] = np.max(dist, axis = tuple(range(1, len(X0.shape)))) - err_vs_dt[:, 2]
        return err_vs_dt
    def get_system_dimension():
        return None

class ODE(base_ODE):
    def __init__(
            self,
            x = None,
            f = None):
        self.x = x
        self.lx = [sp.sympify(fk) for fk in f]
        self.rhs_func = [sp.utilities.lambdify(tuple(self.x), fk,  np)
                           for fk in self.lx]
        return None
    def rhs(self, x):
        return np.array([self.rhs_func[k](*tuple(x))
                         for k in range(len(self.x))])
    def get_system_dimension(self):
        return len(self.x)

class Hamiltonian(ODE):
    def __init__(
            self,
            q = None,
            p = None,
            H = None):
        self.q = q
        self.p = p
        self.H = H
        super(Hamiltonian, self).__init__(
                x = self.q + self.p,
                f = ([ H.diff(pi) for pi in self.p] +
                     [-H.diff(qi) for qi in self.q]))
        self.degrees_of_freedom = len(self.q)
        return None
    def qrhs(self, x):
        return np.array([self.rhs_func[k](*tuple(x))
                         for k in range(self.degrees_of_freedom)])
    def prhs(self, x):
        return np.array([self.rhs_func[k](*tuple(x))
                         for k in range(
                             self.degrees_of_freedom,
                             2*self.degrees_of_freedom)])
    def CM2(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            X[t] = X[t-1]
            X[t, :self.degrees_of_freedom] += 0.5*h*self.qrhs(X[t])
            X[t, self.degrees_of_freedom:] +=     h*self.prhs(X[t])
            X[t, :self.degrees_of_freedom] += 0.5*h*self.qrhs(X[t])
        return X
    def CM4(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        alpha5 =  0.082984406417405
        alpha4 =  0.23399525073150
        alpha3 = -0.40993371990193
        alpha2 =  0.059762097006575
        alpha1 =  0.37087741497958
        alpha0 =  0.16231455076687
        for t in range(1, nsteps + 1):
            X[t] = X[t-1]
            X[t, :self.degrees_of_freedom] += (       alpha5)*h*self.qrhs(X[t])
            X[t, self.degrees_of_freedom:] += (alpha5+alpha0)*h*self.prhs(X[t])
            X[t, :self.degrees_of_freedom] += (alpha0+alpha4)*h*self.qrhs(X[t])
            X[t, self.degrees_of_freedom:] += (alpha4+alpha1)*h*self.prhs(X[t])
            X[t, :self.degrees_of_freedom] += (alpha1+alpha3)*h*self.qrhs(X[t])
            X[t, self.degrees_of_freedom:] += (alpha3+alpha2)*h*self.prhs(X[t])
            X[t, :self.degrees_of_freedom] += (alpha2+alpha2)*h*self.qrhs(X[t])
            X[t, self.degrees_of_freedom:] += (alpha2+alpha3)*h*self.prhs(X[t])
            X[t, :self.degrees_of_freedom] += (alpha3+alpha1)*h*self.qrhs(X[t])
            X[t, self.degrees_of_freedom:] += (alpha1+alpha4)*h*self.prhs(X[t])
            X[t, :self.degrees_of_freedom] += (alpha4+alpha0)*h*self.qrhs(X[t])
            X[t, self.degrees_of_freedom:] += (alpha0+alpha5)*h*self.prhs(X[t])
            X[t, :self.degrees_of_freedom] += (alpha5       )*h*self.qrhs(X[t])
        return X

