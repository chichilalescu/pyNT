import numpy as np

class SDE1D(object):
    def __init__(self):
        return None
    def EM(self, W, X0):
        X = W.W.copy()
        X[0] = X0
        for t in range(1, W.nsteps + 1):
            X[t] = (X[t-1]
                  + self.drift(X[t-1])*W.dt
                  + self.vol(  X[t-1])*(W.W[t] - W.W[t-1]))
        return X

class linSDE(SDE1D):
    def __init__(
            self,
            c0,
            c1):
        self.c0 = c0
        self.c1 = c1
        return None
    def drift(self, x):
        return self.c0*x
    def vol(self, x):
        return self.c1*x
    def exact(self, W, X0):
        time = W.get_time()
        return X0*np.exp(
                    (self.c0 - .5*self.c1**2)*time[:, np.newaxis, np.newaxis]
                    + self.c1*W.W)

class addSDE(SDE1D):
    def __init__(
            self,
            c0,
            c1):
        self.c0 = c0
        self.c1 = c1
        return None
    def drift(self, x):
        return self.c0*x
    def vol(self, x):
        return self.c1

class genericSDE(object):
    def __init__(self):
        return None
    def EM(self, W, X0):
        X = np.zeros(tuple([W[0].W.shape[0]] +
                           list(X0.shape) +
                           list(W[0].W.shape[1:])),
                     X0.dtype)
        for j in range(len(W)):
            assert(W[j].dt == W[0].dt)
            assert(W[j].W.shape == W[0].W.shape)
        X[0] = X0[..., np.newaxis, np.newaxis]
        for t in range(1, W[0].nsteps + 1):
            X[t] = (X[t-1]
                  + self.drift(X[t-1])*W[0].dt)
            b = self.vol(X[t-1])
            for j in range(len(W)):
                X[t] += b[j][..., np.newaxis, np.newaxis]*(W[j].W[t] - W[j].W[t-1])
        return X

class dwell(genericSDE):
    def __init__(
            self,
            c = 0.,
            n = 1.):
        self.c = c
        self.noise = n
        return None
    def get_noise_dimension(
            self):
        return 1
    def get_system_dimension(
            self):
        return 2
    def drift(
            self,
            x):
        return np.array([x[1],
                        -(4*x[0]**3 + 3*self.c*x[0]**2 - 2*x[0] + x[1])])
    def vol(
            self,
            x):
        return [np.array([0, self.noise])]

