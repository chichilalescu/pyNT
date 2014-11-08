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
from wiener import Wiener
import SDE

bla = SDE.linSDE(1.0, 0.5)

bla.get_evdt_vs_M(
            fig_name = 'figs/lin_evdt',
            ntraj = 64,
            X0 = np.array([1.]),
            h0 = .5,
            exp_range = range(8))

bla = SDE.addSDE(1.0, 0.5)

bla.get_evdt_vs_M(
            fig_name = 'figs/add_evdt',
            ntraj = 64,
            X0 = np.array([1.]),
            h0 = .5,
            exp_range = range(8))

bla = SDE.dwell(.0, .5)

bla.get_evdt_vs_M(
            fig_name = 'figs/dwell_evdt',
            ntraj = 64,
            X0 = np.array([0., .0]),
            h0 = 2.**(-3),
            exp_range = range(12))

