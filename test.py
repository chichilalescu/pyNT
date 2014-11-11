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
from sde import sde
from ode import ode

test_sde = False
test_ode = True

if test_sde:
    x = sp.Symbol('x')
    bla = sde(
            x = [x],
            a = [x],
            b = [[x/2, sp.sin(x)/3]])
    bla.get_evdt_vs_M(
            fig_name = 'figs/extra_tst',
            ntraj = 64,
            X0 = np.ones(1,).astype(np.float),
            h0 = .5,
            exp_range = range(8))

    bla = sde(
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
    bla = sde(
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
    bla = ode(
            x = [x, y, z],
            f = lorenz_rhs)
    X0 = 10*np.random.random((3, 128))
    evdt1 = bla.get_evdt(
            X0 = X0,
            solver = ['Euler', 'Taylor2'])
    evdt2 = bla.get_evdt(
            X0 = X0,
            solver = ['Heun', 'Taylor2'])
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111)
    ax.errorbar(
            evdt1[:, 0],
            evdt1[:, 2],
            yerr = [evdt1[:, 1], evdt1[:, 3]],
            label = 'Euler vs Taylor 2')
    ax.errorbar(
            evdt2[:, 0],
            evdt2[:, 2],
            yerr = [evdt2[:, 1], evdt2[:, 3]],
            label = 'Heun vs Taylor 2')
    ax.plot(evdt1[:, 0],
            evdt1[:, 0],
            label = '$\\Delta t$')
    ax.plot(evdt1[:, 0],
            evdt1[:, 0]**2,
            label = '$\\Delta t^2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    fig.savefig('figs/ode_evdt.pdf', format = 'pdf')

