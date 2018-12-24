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

class LyapunovMRAC():
    """LyapunovMRAC

    Attributes
    -----------
    input : float
        system state, this system should have one input - one output
    a : float
        parameter of reference model
    alpha_1 : float
        parameter of the controller
    alpha_2 : float
        parameter of the controller
    theta_1 : float
        state of the controller
    theta_2 : float
        state of the controller
    history_input : list
        time history of input
    """
    def __init__(self, g_1, g_2, init_theta_1=0.0, init_theta_2=0.0, init_input=0.0):
        """
        Parameters
        -----------
        g_1 : float
            parameter of the controller
        g_2 : float
            parameter of the controller
        theta_1 : float, optional
            state of the controller default is 0.0
        theta_2 : float, optional
            state of the controller default is 0.0
        init_input : float, optional
            initial input of controller default is 0.0
        """
        self.input = init_input

        # parameters
        self.g_1 = g_1
        self.g_2 = g_2

        # states
        self.theta_1 = init_theta_1
        self.theta_2 = init_theta_2

        self.history_input = [init_input]

    def update_input(self, e, r, y, dt=0.01):
        """
        Parameters
        ------------
        e : float
            error value of system
        r : float
            reference value
        y : float
            output the model value
        dt : float in seconds, optional
            sampling time of simulation, default is 0.01 [s]
        """
        # for theta 1, theta 1 dot, theta 2, theta 2 dot
        k0 = [0.0 for _ in range(4)]
        k1 = [0.0 for _ in range(4)]
        k2 = [0.0 for _ in range(4)]
        k3 = [0.0 for _ in range(4)]

        functions = [self._func_theta_1, self._func_theta_2]

        # solve Runge-Kutta
        for i, func in enumerate(functions):
            k0[i] = dt * func(self.theta_1, self.theta_2, e, r, y)
        
        for i, func in enumerate(functions):
            k1[i] = dt * func(self.theta_1 + k0[0]/2.0, self.theta_2 + k0[1]/2.0, e, r, y)
        
        for i, func in enumerate(functions):
            k2[i] = dt * func(self.theta_1 + k1[0]/2.0, self.theta_2 + k1[1]/2.0, e, r, y)
        
        for i, func in enumerate(functions):
            k3[i] = dt * func(self.theta_1 + k2[0], self.theta_2 + k2[1], e, r, y)
        
        self.theta_1 += (k0[0] + 2 * k1[0] + 2 * k2[0] + k3[0]) / 6.0
        self.theta_2 += (k0[1] + 2 * k1[1] + 2 * k2[1] + k3[1]) / 6.0

        # for oylar
        """
        self.theta_1 += k0[0]
        self.u_1 += k0[1]
        self.theta_2 += k0[2]
        self.u_2 += k0[3]
        """
        # calc input
        self.input = self.theta_1 * r + self.theta_2 * y

        # save
        self.history_input.append(self.input)

    def _func_theta_1(self, theta_1, theta_2, e, r, y):
        """
        Parameters
        ------------
        theta_1 : float
            state of the controller
        theta_2 : float
            state of the controller
        e : float
            error
        r : float
            reference
        y : float
            output of system 
        """
        y_dot = self.g_1 * r * e

        return y_dot

    def _func_theta_2(self, theta_1, theta_2, e, r, y):
        """
        Parameters
        ------------
        Parameters
        ------------
        theta_1 : float
            state of the controller
        theta_2 : float
            state of the controller
        e : float
            error
        r : float
            reference
        y : float
            output of system 
        """
        y_dot = self.g_2 * y * e

        return y_dot
    
    
def main():
    # control plant
    a = -0.5
    b = 0.5
    plant = FirstOrderSystem(a, b)

    # reference model
    a = 1.
    b = 1.
    reference_model = FirstOrderSystem(a, b)

    # controller
    g_1 = 5.
    g_2 = 5.
    controller = LyapunovMRAC(g_1, g_2)

    simulation_time = 50 # in second
    dt = 0.01
    simulation_iterations = int(simulation_time / dt) # dt is 0.01

    history_error = [0.0]
    history_r = [0.0]

    for i in range(1, simulation_iterations): # skip the first
        # reference input
        r = math.sin(dt * i)
        # update reference
        reference_model.update_state(r, dt=dt)
        # update plant
        plant.update_state(controller.input, dt=dt)

        # calc error
        e = reference_model.state - plant.state
        y = plant.state
        history_error.append(e)
        history_r.append(r)

        # make the gradient
        controller.update_input(e, r, y, dt=dt)

    # fig
    plt.plot(np.arange(simulation_iterations)*dt, plant.history_state, label="plant y", linestyle="dashed")
    plt.plot(np.arange(simulation_iterations)*dt, reference_model.history_state, label="model reference")
    plt.plot(np.arange(simulation_iterations)*dt, history_error, label="error", linestyle="dashdot")
    # plt.plot(range(simulation_iterations), history_r, label="error")
    plt.xlabel("time [s]")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # input
    # plt.plot(np.arange(simulation_iterations)*dt, controller.history_input)
    # plt.show()

if __name__ == "__main__":
    main()