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
    def EE2(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            X1 = X[t-1] + self.rhs(X[t-1])*h
            X2 = X[t-1] + self.rhs(X[t-1])*h*0.5
            X2 += self.rhs(X2)*h*0.5
            X[t] = 2*X2 - X1
        return X
    def EE3(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            F = self.rhs(X[t-1])
            X1 = X[t-1] + F*h
            X2 = X[t-1] + F*h*0.5
            X2 += self.rhs(X2)*h*0.5
            X3 = X[t-1] + F*h/3
            X3 += self.rhs(X3)*h/3
            X3 += self.rhs(X3)*h/3
            X[t] = 4.5*X3 - 4*X2 + 0.5*X1
        return X
    def EE4(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            F = self.rhs(X[t-1])
            X1 = X[t-1] + F*h
            X2 = X[t-1] + F*h*0.5
            X2 += self.rhs(X2)*h*0.5
            X3 = X[t-1] + F*h/3
            X3 += self.rhs(X3)*h/3
            X3 += self.rhs(X3)*h/3
            X4 = X[t-1] + F*h/4
            X4 += self.rhs(X4)*h/4
            X4 += self.rhs(X4)*h/4
            X4 += self.rhs(X4)*h/4
            X[t] = 32*X4/3. - 13.5*X3 + 4*X2 - X1/6
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
        x0 = [getattr(self, solver[0])(h0*2.**(-n), nsteps * 2**n, X0)
                for n in exp_range]
        x1 = [getattr(self, solver[1])(h0*2.**(-n), nsteps * 2**n, X0)
                for n in exp_range]
        dist = [np.average(np.abs(x0[p][-1] - x1[p][-1]), axis = 0)
                for p in range(len(x0))]
        if relative:
            for p in range(len(x0)):
                dist[p] /= np.average(np.abs(x0[p][-1]), axis = 0)
        err_vs_dt = np.zeros((len(x0), 4), dtype = X0.dtype)
        err_vs_dt[:, 0] = np.array([h0*2.**(-n) for n in exp_range])
        err_vs_dt[:, 2] = np.average(
            dist,
            axis = tuple(range(1, len(X0.shape))))
        err_vs_dt[:, 1] = (err_vs_dt[:, 2] -
                           np.min(dist, axis = tuple(range(1, len(X0.shape)))))
        err_vs_dt[:, 3] = (np.max(dist, axis = tuple(range(1, len(X0.shape)))) -
                           err_vs_dt[:, 2])
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
        # CM2 coefficients
        self.CM2_coeffs = [0.5, 1.0, 0.5]
        # CM4 coefficients
        alpha5 =  0.082984406417405
        alpha4 =  0.23399525073150
        alpha3 = -0.40993371990193
        alpha2 =  0.059762097006575
        alpha1 =  0.37087741497958
        alpha0 =  0.16231455076687
        self.CM4_coeffs = [       alpha5,
                           alpha5+alpha0,
                           alpha0+alpha4,
                           alpha4+alpha1,
                           alpha1+alpha3,
                           alpha3+alpha2,
                           alpha2+alpha2,
                           alpha2+alpha3,
                           alpha3+alpha1,
                           alpha1+alpha4,
                           alpha4+alpha0,
                           alpha0+alpha5,
                           alpha5       ]
        # CM6 coefficients
        gamma1 =  0.39216144400731413927925056
        gamma2 =  0.33259913678935943859974864
        gamma3 = -0.70624617255763935980996482
        gamma4 =  0.08221359629355080023149045
        gamma5 =  0.79854399093482996339895035
        self.CM6_coeffs = [
            gamma1*.5,
            gamma1,
            (gamma1 + gamma2)*.5,
            gamma2,
            (gamma2 + gamma3)*.5,
            gamma3,
            (gamma3 + gamma4)*.5,
            gamma4,
            (gamma4 + gamma5)*.5,
            gamma5,
            (gamma5 + gamma4)*.5,
            gamma4,
            (gamma4 + gamma3)*.5,
            gamma3,
            (gamma3 + gamma2)*.5,
            gamma2,
            (gamma2 + gamma1)*.5,
            gamma1,
            gamma1*.5]
        # CM8 coefficients
        gamma1 =  0.13020248308889008087881763
        gamma2 =  0.56116298177510838456196441
        gamma3 = -0.38947496264484728640807860
        gamma4 =  0.15884190655515560089621075
        gamma5 = -0.39590389413323757733623154
        gamma6 =  0.18453964097831570709183254
        gamma7 =  0.25837438768632204729397911
        gamma8 =  0.29501172360931029887096624
        gamma9 = -0.60550853383003451169892108
        self.CM8_coeffs = [(gamma1*0.5),
                           (gamma1),
                           ((gamma1+gamma2)*0.5),
                           (gamma2),
                           ((gamma2+gamma3)*0.5),
                           (gamma3),
                           ((gamma3+gamma4)*0.5),
                           (gamma4),
                           ((gamma4+gamma5)*0.5),
                           (gamma5),
                           ((gamma5+gamma6)*0.5),
                           (gamma6),
                           ((gamma6+gamma7)*0.5),
                           (gamma7),
                           ((gamma7+gamma8)*0.5),
                           (gamma8),
                           ((gamma8+gamma9)*0.5),
                           (gamma9),
                           ((gamma9+gamma8)*0.5),
                           (gamma8),
                           ((gamma8+gamma7)*0.5),
                           (gamma7),
                           ((gamma7+gamma6)*0.5),
                           (gamma6),
                           ((gamma6+gamma5)*0.5),
                           (gamma5),
                           ((gamma5+gamma4)*0.5),
                           (gamma4),
                           ((gamma4+gamma3)*0.5),
                           (gamma3),
                           ((gamma3+gamma2)*0.5),
                           (gamma2),
                           ((gamma2+gamma1)*0.5),
                           (gamma1),
                           (gamma1*0.5)]
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
        for t in range(1, nsteps + 1):
            X[t] = X[t-1]
            X[t, :self.degrees_of_freedom] += (
                self.CM4_coeffs[0]*h*self.qrhs(X[t]))
            for i in range(1, len(self.CM4_coeffs), 2):
                X[t, self.degrees_of_freedom:] += (
                    self.CM4_coeffs[i  ]*h*self.prhs(X[t]))
                X[t, :self.degrees_of_freedom] += (
                    self.CM4_coeffs[i+1]*h*self.qrhs(X[t]))
        return X
    def CM6(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            X[t] = X[t-1]
            X[t, :self.degrees_of_freedom] += (
                self.CM6_coeffs[0]*h*self.qrhs(X[t]))
            for i in range(1, len(self.CM6_coeffs), 2):
                X[t, self.degrees_of_freedom:] += (
                    self.CM6_coeffs[i  ]*h*self.prhs(X[t]))
                X[t, :self.degrees_of_freedom] += (
                    self.CM6_coeffs[i+1]*h*self.qrhs(X[t]))
        return X
    def CM8(self, h, nsteps, X0):
        X = np.zeros((nsteps+1,) + X0.shape,
                     X0.dtype)
        X[0, :] = X0
        for t in range(1, nsteps + 1):
            X[t] = X[t-1]
            X[t, :self.degrees_of_freedom] += self.CM8_coeffs[0]*h*self.qrhs(X[t])
            for i in range(1, len(self.CM8_coeffs), 2):
                X[t, self.degrees_of_freedom:] += self.CM8_coeffs[i  ]*h*self.prhs(X[t])
                X[t, :self.degrees_of_freedom] += self.CM8_coeffs[i+1]*h*self.qrhs(X[t])
        return X

