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

system = SDE.dwell(0, .5)

T = 10.
h0 = 1.
substeps = 2**10

bla1 = [Wiener(
        nsteps = int(T / h0)*substeps,
        dt = h0 / substeps,
        nbatches = 20,
        ntraj = 128)]
bla1[0].initialize()
x = system.EM(bla1, np.array([.0, .0]))

fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111)
ax.hist(x[substeps*2:, 0].flatten(),
        histtype = 'step',
        bins = 128,
        normed = True,
        cumulative = True)
ax.grid()
fig.savefig('tst.pdf', format = 'pdf')

