import numpy as np
from wiener import Wiener
import SDE

bla = SDE.linSDE(1.0, 0.5)

bla.get_evdt_vs_M(
            fig_name = 'lin_evdt',
            ntraj = 64,
            X0 = np.array([1.]),
            h0 = .5,
            exp_range = range(8))

bla = SDE.addSDE(1.0, 0.5)

bla.get_evdt_vs_M(
            fig_name = 'add_evdt',
            ntraj = 64,
            X0 = np.array([1.]),
            h0 = .5,
            exp_range = range(8))

