import numpy as np
import matplotlib.pyplot as plt
import math

class FirstOrderSystem():
    """FirstOrderSystem

    Attributes
    -----------
    state : float
        system state, this system should have one input - one output
    a : float
        parameter of the system
    b : float
        parameter of the system
    history_state : list
        time history of state
    """
    def __init__(self, a, b, init_state=0.0):
        """
        Parameters
        -----------
        a : float
            parameter of the system
        b : float
            parameter of the system
        init_state : float, optional
            initial state of system default is 0.0
        """
        self.state = init_state
        self.a = a
        self.b = b
        self.history_state = [init_state]

    def update_state(self, u, dt=0.01):
        """calculating input
        Parameters
        ------------
        u : float
            input of system in some cases this means the reference
        dt : float in seconds, optional
            sampling time of simulation, default is 0.01 [s]
        """
        # solve Runge-Kutta
        k0 = dt * self._func(self.state, u)
        k1 = dt * self._func(self.state + k0/2.0, u)
        k2 = dt * self._func(self.state + k1/2.0, u)
        k3 = dt * self._func(self.state + k2, u)

        self.state +=  (k0 + 2 * k1 + 2 * k2 + k3) / 6.0

        # for oylar
        # self.state += k0

        # save
        self.history_state.append(self.state)

    def _func(self, y, u):
        """
        Parameters
        ------------
        y : float
            state of system
        u : float
            input of system in some cases this means the reference
        """
        y_dot = -self.a * y + self.b * u

        return y_dot

