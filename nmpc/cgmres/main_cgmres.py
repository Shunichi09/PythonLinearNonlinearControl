import numpy as np
import matplotlib.pyplot as plt
import math

class SampleSystem():
    """SampleSystem

    Attributes
    -----------
    
    """
    def __init__(self, init_x_1=0., init_x_2=0.):
        """
        Parameters
        -----------
        
        """
        self.x_1 = init_x_1
        self.x_2 = init_x_2
        self.history_x_1 = [init_x_1]
        self.history_x_2 = [init_x_2]

    def update_state(self, u, dt=0.01):
        """
        Parameters
        ------------
        u : float
            input of system in some cases this means the reference
        dt : float in seconds, optional
            sampling time of simulation, default is 0.01 [s]
        """
        # for theta 1, theta 1 dot, theta 2, theta 2 dot
        k0 = [0.0 for _ in range(2)]
        k1 = [0.0 for _ in range(2)]
        k2 = [0.0 for _ in range(2)]
        k3 = [0.0 for _ in range(2)]

        functions = [self._func_x_1, self._func_x_2]

        # solve Runge-Kutta
        for i, func in enumerate(functions):
            k0[i] = dt * func(self.x_1, self.x_2, u)
        
        for i, func in enumerate(functions):
            k1[i] = dt * func(self.x_1 + k0[0]/2., self.x_2 + k0[1]/2., u)
        
        for i, func in enumerate(functions):
            k2[i] = dt * func(self.x_1 + k1[0]/2., self.x_2 + k1[1]/2., u)
        
        for i, func in enumerate(functions):
            k3[i] =  dt * func(self.x_1 + k2[0], self.x_2 + k2[1], u)
        
        self.x_1 += (k0[0] + 2. * k1[0] + 2. * k2[0] + k3[0]) / 6.
        self.x_2 += (k0[1] + 2. * k1[1] + 2. * k2[1] + k3[1]) / 6.
    
        # save
        self.history_x_1.append(self.x_1)
        self.history_x_2.append(self.x_2)

    def _func_x_1(self, y_1, y_2, u):
        """
        Parameters
        ------------
        
        """
        y_dot = y_2

        return y_dot
    
    def _func_x_2(self, y_1, y_2, u):
        """
        Parameters
        ------------
        
        """
        y_dot = (1 - y_1**2 - y_2**2) * y_2 - y_1 + u

        return y_dot


def main():
    # simulation time
    dt = 0.01
    iteration_time = 20.
    iteration_num = int(iteration_time/dt)

    # plant
    plant_system = SampleSystem(init_x_1=2., init_x_2=0.)

    # controller

    for i in range(iteration_num):
        u = 1.0
        plant_system.update_state(u)
    
    # figure
    fig = plt.figure()

    x_1_fig = fig.add_subplot(231)
    x_2_fig = fig.add_subplot(232)
    u_fig = fig.add_subplot(233)

    x_1_fig.plot(np.arange(iteration_num+1)*dt, plant_system.history_x_1)
    x_2_fig.plot(np.arange(iteration_num+1)*dt, plant_system.history_x_2)

    plt.show()


if __name__ == "__main__":
    main()


    
