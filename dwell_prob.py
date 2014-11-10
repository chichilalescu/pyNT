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
from sde import sde

def get_probability(
        c = 0.0,
        T = 10.,
        h0 = 1.,
        substeps = 2**10,
        nbatches = 10,
        ntraj = 64,
        figname = 'dwell_PDF'):
    x = sp.Symbol('x')
    v = sp.Symbol('v')
    u = x**2 * (x**2 - 1 + c*x)
    system = sde(
            x = [x, v],
            a = [v, -u.diff(x) - v],
            b = [[.0], [.5]])

    bla1 = Wiener(
            nsteps = int(T / h0)*substeps,
            dt = h0 / substeps,
            noise_dimension = 1,
            solution_shape = [nbatches, ntraj])
    bla1.initialize()
    x = system.EM(bla1, np.array([.0, .0]))

    points = x[substeps*2:, 0].flatten()

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)
    hist, bins, bla = ax.hist(points,
            histtype = 'step',
            bins = 128,
            normed = True)
    probability = np.count_nonzero(points > 0) * (1./ points.size)
    fig.suptitle(
            'Probability density function for c = {0}.'.format(c))
    ax.set_title('$\\lim_{{t \\rightarrow \\infty}} P[X_t > 0] \\approx {0:.2g}$'.format(probability))
    fig.savefig(figname + '.pdf', format = 'pdf')
    return probability, hist, bins

p00, h00, b00 = get_probability(
        c = 0.0,
        figname = 'figs/dwell00_PDF')
p03, h03, b03 = get_probability(
        c = 0.3,
        figname = 'figs/dwell03_PDF')
p06, h06, b06 = get_probability(
        c = 0.6,
        figname = 'figs/dwell06_PDF')
p10, h10, b10 = get_probability(
        c = 1.0,
        figname = 'figs/dwell10_PDF')

fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111)
ax.plot(.5*(b00[1:] + b00[:-1]),
        h00,
        label = '$c = 0.0$, $\\lim_{{t \\rightarrow \\infty}} P[X_t > 0] \\approx {0:.2g}$'.format(p00))
ax.plot(.5*(b03[1:] + b03[:-1]),
        h03,
        label = '$c = 0.3$, $\\lim_{{t \\rightarrow \\infty}} P[X_t > 0] \\approx {0:.2g}$'.format(p03))
ax.plot(.5*(b03[1:] + b06[:-1]),
        h06,
        label = '$c = 0.6$, $\\lim_{{t \\rightarrow \\infty}} P[X_t > 0] \\approx {0:.2g}$'.format(p06))
ax.plot(.5*(b10[1:] + b10[:-1]),
        h10,
        label = '$c = 1.0$, $\\lim_{{t \\rightarrow \\infty}} P[X_t > 0] \\approx {0:.2g}$'.format(p10))
ax.legend(loc = 'best')
fig.savefig('figs/dwell_PDF.pdf', format = 'pdf')

