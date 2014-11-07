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

t1ma_nm1 = {'0.90,009': 1.83,
            '0.90,019': 1.73,
            '0.90,029': 1.70,
            '0.90,039': 1.68,
            '0.90,059': 1.67,
            '0.90,099': 1.66,
            '0.90,199': 1.65,
            '0.99,009': 3.25,
            '0.99,019': 2.86,
            '0.99,029': 2.76,
            '0.99,039': 2.70,
            '0.99,059': 2.66,
            '0.99,099': 2.62,
            '0.99,199': 2.58}

def get_t1ma_nm1(
        onema,
        nm1):
    return t1ma_nm1['{0:.2f},{1:0>3}'.format(onema, nm1)]

class Wiener:
    def __init__(
            self,
            dt = 1.,
            nsteps = 128,
            nbatches = 20,
            ntraj = 16):
        self.dt = dt
        self.nsteps = nsteps
        self.nbatches = nbatches
        self.ntraj = ntraj
        return None
    def initialize(
            self,
            rseed = None):
        np.random.seed(rseed)
        self.dW = np.sqrt(self.dt)*np.random.randn(self.nsteps, self.nbatches, self.ntraj)
        self.W = np.zeros(
                (self.nsteps + 1, self.nbatches, self.ntraj),
                dtype = self.dW.dtype)
        for t in range(self.nsteps):
            self.W[t+1] = self.W[t] + self.dW[t]
        return None
    def get_time(
            self):
        return self.dt*np.array(range(self.W.shape[0]))
    def coarsen(
            self,
            n = 2):
        new_object = Wiener(
                dt = n*self.dt,
                nsteps = self.nsteps/n,
                nbatches = self.nbatches,
                ntraj = self.ntraj)
        new_object.W = self.W[::n]
        return new_object

