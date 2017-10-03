import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d

class DDE_state:

    def __init__(self):
        self.N = lambda x: np.exp(-x**2)
        self.p = lambda x: x
        self.f = np.vectorize(lambda x: self.N(self.p(x)))

    def set_price_function(self,p):
        self.p = np.vectorize(p)
        self.f = np.vectorize(lambda x: self.N(self.p(x)))

    def set_spending_ability(self,N):
        self.N = np.vectorize(N)
        self.f = np.vectorize(lambda x: self.N(self.p(x)))

    def evolve(self,ts,xs):
        f = self.f
        xs_old = interp1d(ts,xs,fill_value=(xs[0],xs[-1]),bounds_error=False)
        def rhs(t,x):
            return [f(x[0])-f(xs_old(t))]
        r = ode(rhs).set_integrator('vode', method='bdf')
        r.set_initial_value([xs[-1]],ts[0])
        res = [xs[-1]]+[ r.integrate(t)[0] for t in ts[1:]]
        return res

    def consistent_x0s(self,ts):
        f = self.f
        def rhs(t,x):
            return [f(x[0])]
        r = ode(rhs).set_integrator('vode', method='bdf')
        r.set_initial_value(0,ts[0])
        res = [0]+[ r.integrate(t)[0] for t in ts[1:]]
        return res

    def evolve_n(self,ts,x0s,Nmax):
        f = self.f
        inds = np.arange(Nmax)
        res = [x0s]
        for i in inds:
            res.append(self.evolve(ts,res[-1]))
        return res

    def consistent_evolve(self,ts,Nmax):
        x0s = self.consistent_x0s(ts)
        return self.evolve_n(ts,x0s,Nmax)
