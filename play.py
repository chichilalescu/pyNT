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
import matplotlib.pyplot as plt
from wiener import Wiener, get_t1ma_nm1
import SDE

def get_evdt_vs_M(
        my_system,
        fig_name = 'tst',
        ntraj = 32,
        X0 = None,
        h0 = 2.**(-3),
        exp_range = range(8)):
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_axes([.1, .1, .8, .8])
    for M in [10, 20, 30, 40, 60, 100, 200]:
        if issubclass(type(my_system), SDE.SDE1D):
            bla1 = Wiener(
                    nsteps = 2**8,
                    dt = h0 / (2**8),
                    nbatches = M,
                    ntraj = ntraj)
            bla1.initialize(rseed=5)
            wiener_paths = [bla1.coarsen(2**n) for n in exp_range]
            if type(X0) == type(None):
                X0 = 1.
            dtlist = [wiener_paths[p].dt for p in range(len(wiener_paths))]
        else:
            bla1 = [Wiener(
                    nsteps = 2**8,
                    dt = h0 / (2**8),
                    nbatches = M,
                    ntraj = ntraj) for j in range(my_system.get_noise_dimension())]
            for bla in bla1:
                bla.initialize()
            wiener_paths = [[bla.coarsen(2**n)
                             for bla in bla1]
                            for n in exp_range]
            if type(X0) == type(None):
                X0 = np.zeros(my_system.get_system_dimension(), dtype = np.float)
            dtlist = [wiener_paths[p][0].dt for p in range(len(wiener_paths))]
        xnumeric = [my_system.EM(wiener_paths[p], X0) for p in range(len(wiener_paths))]
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
    ax.plot(dtlist, dtlist, label = '$\\Delta t$')
    ax.plot(dtlist, np.array(dtlist)**.5, label = '$\\Delta t^{1/2}$')
    ax.set_xlabel('$\\Delta t$')
    ax.set_ylabel('$\\epsilon$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    fig.savefig(fig_name + '.pdf', format = 'pdf')
    return None

system = SDE.dwell(0, .5)

#get_evdt_vs_M(
#        system,
#        ntraj = 128,
#        X0 = np.array([0, 0]).astype(np.float),
#        h0 = 1.,
#        exp_range = range(10))

T = 10.
h0 = 1.
substeps = 2**14

bla1 = [Wiener(
        nsteps = int(T / h0)*substeps,
        dt = h0 / substeps,
        nbatches = 1,
        ntraj = 32)]
bla1[0].initialize()
x = system.EM(bla1, np.array([.0, .0]))

fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(111)
ax.plot(x[:, 0, 0])
fig.savefig('tst.pdf', format = 'pdf')
