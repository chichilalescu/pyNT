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

from wiener import Wiener
from sde import SDE
from ode import ODE

test_sde = True
test_ode = False

if test_sde:
    x = sp.Symbol('x')
    bla = SDE(
            x = [x],
            a = [x],
            b = [[x/2, sp.sin(x)/3]])
    bla.get_evdt_vs_M(
            fig_name = 'figs/extra_tst',
            ntraj = 64,
            X0 = np.ones(1,).astype(np.float),
            h0 = .5,
            exp_range = range(8))

    bla = SDE(
            x = [x],
            a = [x],
            b = [[.5, .25]])
    bla.get_evdt_vs_M(
            fig_name = 'figs/add_evdt',
            ntraj = 64,
            X0 = np.ones(1,).astype(np.float),
            h0 = .5,
            exp_range = range(8),
            solver = ['Taylor_2p0_additive', 'explicit_1p5_additive'])

    c = 0.0
    v = sp.Symbol('v')
    u = x**2 * (x**2 - 1 + c*x)
    bla = SDE(
            x = [x, v],
            a = [v, -u.diff(x) - v],
            b = [[.01, .1], [.95, .3]])
    bla.get_evdt_vs_M(
            fig_name = 'figs/dwell_evdt',
            ntraj = 32,
            X0 = np.array([0., .0]).astype(np.float),
            h0 = .5,
            solver = ['Taylor_2p0_additive', 'explicit_1p5_additive'],
            exp_range = range(8))

if test_ode:
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    z = sp.Symbol('z')
    rho = 28.
    sigma = 10.
    beta = 8./3
    lorenz_rhs = [sigma*(y - x),
                  x*(rho - z) - y,
                  x*y - beta*z]
    bla = ODE(
            x = [x, y, z],
            f = lorenz_rhs)
    X0 = 10*np.random.random((3, 128))
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111)
    for solver in [['Euler', 'Taylor2'],
                   ['Heun',  'Taylor2'],
                   ['cRK',   'Taylor4']]:
        evdt = bla.get_evdt(
                X0 = X0,
                solver = solver)
        ax.errorbar(
                evdt[:, 0],
                evdt[:, 2],
                yerr = [evdt[:, 1], evdt[:, 3]],
                label = '{0} vs {1}'.format(solver[0], solver[1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    fig.savefig('figs/ode_evdt.pdf', format = 'pdf')

