import numpy as np
import matplotlib.pyplot as plt
import math
import copy


"""
このWheeled modelはコントローラー用
ホントはbase作って、継承すべきですが省略
"""
class TwoWheeledCar():
    """SampleSystem, this is the simulator
    Attributes
    -----------
    xs : numpy.ndarray
        system states, [x, y, theta]
    history_xs : list
        time history of state
    """
    def __init__(self, init_states=None):
        """
        Palameters
        -----------
        init_state : float, optional, shape(3, )
            initial state of system default is None
        """
        self.STATE_SIZE = 3
        self.INPUT_SIZE = 2

        self.xs = np.zeros(3)
    
        if init_states is not None:
            self.xs = copy.deepcopy(init_states)

        self.history_xs = [init_states]
        self.history_predict_xs = []

    def update_state(self, us, dt):
        """
        Palameters
        ------------
        us : numpy.ndarray
            inputs of system in some cases this means the reference
        dt : float in seconds, optional
            sampling time of simulation, default is 0.01 [s]
        """
        # for theta 1, theta 1 dot, theta 2, theta 2 dot
        k0 = [0.0 for _ in range(3)]
        k1 = [0.0 for _ in range(3)]
        k2 = [0.0 for _ in range(3)]
        k3 = [0.0 for _ in range(3)]

        functions = [self._func_x_1, self._func_x_2, self._func_x_3]

        # solve Runge-Kutta
        for i, func in enumerate(functions):
            k0[i] = dt * func(self.xs[0], self.xs[1], self.xs[2], us[0], us[1])

        for i, func in enumerate(functions):
            k1[i] = dt * func(self.xs[0] + k0[0]/2., self.xs[1] + k0[1]/2., self.xs[2] + k0[2]/2., us[0], us[1])
        
        for i, func in enumerate(functions):
            k2[i] = dt * func(self.xs[0] + k0[0]/2., self.xs[1] + k0[1]/2., self.xs[2] + k0[2]/2., us[0], us[1])
        
        for i, func in enumerate(functions):
            k3[i] =  dt * func(self.xs[0] + k2[0], self.xs[1] + k2[1], self.xs[2] + k2[2], us[0], us[1])
        
        self.xs[0] += (k0[0] + 2. * k1[0] + 2. * k2[0] + k3[0]) / 6.
        self.xs[1] += (k0[1] + 2. * k1[1] + 2. * k2[1] + k3[1]) / 6.
        self.xs[2] += (k0[2] + 2. * k1[2] + 2. * k2[2] + k3[2]) / 6.
    
        # save
        save_states = copy.deepcopy(self.xs)
        self.history_xs.append(save_states)

        return self.xs.copy()
    
    def initialize_state(self, init_xs):
        """
        initialize the state
        
        Parameters
        ------------
        init_xs : numpy.ndarray
        """
        self.xs = init_xs.flatten()

    def _func_x_1(self, y_1, y_2, y_3, u_1, u_2):
        """
        Parameters
        ------------
        y_1 : float
        y_2 : float
        y_3 : float
        u_1 : float
            system input
        u_2 : float
            system input
        """
        y_dot = math.cos(y_3) * u_1
        return y_dot
    
    def _func_x_2(self, y_1, y_2, y_3, u_1, u_2):
        """
        Parameters
        ------------
        y_1 : float
        y_2 : float
        y_3 : float
        u_1 : float
            system input
        u_2 : float
            system input
        """
        y_dot = math.sin(y_3) * u_1
        return y_dot
    
    def _func_x_3(self, y_1, y_2, y_3, u_1, u_2):
        """
        Parameters
        ------------
        y_1 : float
        y_2 : float
        y_3 : float
        u_1 : float
            system input
        u_2 : float
            system input
        """
        y_dot = u_2
        return y_dot