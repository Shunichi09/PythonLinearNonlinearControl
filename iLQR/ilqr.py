import numpy as np
from copy import copy, deepcopy

from model import TwoWheeledCar

class iLQRController():
    """
    A controller that implements iterative Linear Quadratic control.
    Controls the (x, y, th) of the two wheeled car

    Attributes:
    ------------
    tN : int
        number of time step
    STATE_SIZE : int
        system state size
    INPUT_SIZE : int
        system input size
    dt : float
        sampling time
    max_iter : int
        number of max iteration
    lamb_factor : int
        lambda factor is that the adding value to make the matrix of Q _uu be positive
    lamb_max : float
        maximum lambda value to make the matrix of Q _uu be positive
    eps_converge : float
        threshold value of the iteration
    """

    def __init__(self, N=100, max_iter=400, dt=0.01): 
        """
        Parameters
        ----------
        N : int, optional
            number of time step, default is 100
        max_iter : int, optional
            number of max iteration, default is 400
        dt : float, optional
            sampling time, default is 0.01
        """
        self.tN = N
        self.STATE_SIZE = 3
        self.INPUT_SIZE = 2
        self.dt = dt

        self.max_iter = max_iter
        self.lamb_factor = 10
        self.lamb_max = 1e4
        self.eps_converge = 0.001

    def calc_input(self, car, x_target):
        """main loop of iterative LQR
        
        Parameters
        -------------
        car : model class
            should have initialize state and update state
        x_target : numpy.ndarray, shape(STATE_SIZE, )
            target state

        Returns
        -----------
        u : numpy.ndarray, shape(INPUT_SIZE, )

        See also
        ----------
        model.py
        """

        # initialize
        self.reset(x_target)

        # Compute the optimization
        x0 = np.zeros(self.STATE_SIZE)
        self.simulator, x0 = self.initialize_simulator(car)
        U = np.copy(self.U)
        self.X, self.U, cost = self.ilqr(x0, U)

        self.u = self.U[self.t] # use only one time step (like MPC)

        return self.u

    def initialize_simulator(self, car):
        """ make copy for controller
        Parameters
        -------------
        car : model class
            should have initialize state and update state

        Returns
        ----------
        simulator : model class
            should have initialize state and update state
        x0 : numpy.ndarray, shape(STATE_SIZE)
            initial state of the simulator
        """ 
        # copy
        simulator = TwoWheeledCar(deepcopy(car.xs))

        return simulator, deepcopy(simulator.xs)

    def cost(self, xs, us):
        """ the immediate state cost function
        
        Parameters
        ------------
        xs : shape(STATE_SIZE, tN + 1)
            predict state of the system
        us : shape(STATE_SIZE, tN)
            predict input of the system

        Returns
        ----------
        l : float 
            stage cost
        l_x : numpy.ndarray, shape(STATE_SIZE, )
            differential of stage cost by x
        l_xx : numpy.ndarray, shape(STATE_SIZE, STATE_SIZE) 
            second order differential of stage cost by x
        l_u : numpy.ndarray, shape(INPUT_SIZE, )
            differential of stage cost by u
        l_uu : numpy.ndarray, shape(INPUT_SIZE, INPUT_SIZE)
            second order differential of stage cost by uu
        l_ux numpy.ndarray, shape(INPUT_SIZE, STATE_SIZE)
            second order differential of stage cost by ux
        """

        # total cost
        R_11 = 0.01 # terminal u_v cost weight
        R_22 = 0.01 # terminal u_th cost weight

        l = np.dot(us.T, np.dot(np.diag([R_11, R_22]), us))

        # compute derivatives of cost
        l_x = np.zeros(self.STATE_SIZE)
        l_xx = np.zeros((self.STATE_SIZE, self.STATE_SIZE))

        l_u1 = 2. * us[0] * R_11
        l_u2 = 2. * us[1] * R_22

        l_u = np.array([l_u1, l_u2])

        l_uu = 2. * np.diag([R_11, R_22])
        
        l_ux = np.zeros((self.INPUT_SIZE, self.STATE_SIZE))

        return l, l_x, l_xx, l_u, l_uu, l_ux

    def cost_final(self, x):
        """ the final state cost function
        
        Parameters
        -------------
        x : numpy.ndarray, shape(STATE_SIZE,)
            predict state of the system

        Returns
        ---------
        l : float
            terminal cost
        l_x : numpy.ndarray, shape(STATE_SIZE, )
            differential of stage cost by x
        l_xx : numpy.ndarray, shape(STATE_SIZE, STATE_SIZE) 
            second order differential of stage cost by x
        """
        Q_11 = 1. # terminal x cost weight
        Q_22 = 1. # terminal y cost weight
        Q_33 = 0.01 # terminal theta cost weight

        error = self.simulator.xs - self.target

        l = np.dot(error.T, np.dot(np.diag([Q_11, Q_22, Q_33]), error))
        
        # about L_x
        l_x1 = 2. * (x[0] - self.target[0]) * Q_11
        l_x2 = 2. * (x[1] - self.target[1]) * Q_22
        l_x3 = 2. * (x[2] -self.target[2]) * Q_33
        l_x = np.array([l_x1, l_x2, l_x3])

        # about l_xx
        l_xx =  2. * np.diag([Q_11, Q_22, Q_33])

        # Final cost only requires these three values
        return l, l_x, l_xx

    def finite_differences(self, x, u): 
        """ calculate gradient of plant dynamics using finite differences
        
        Parameters
        --------------
        x : numpy.ndarray, shape(STATE_SIZE,)
            the state of the system
        u : numpy.ndarray, shape(INPUT_SIZE,)
            the control input

        Returns
        ------------
        A : numpy.ndarray, shape(STATE_SIZE, STATE_SIZE)
            differential of the model /alpha X
        B : numpy.ndarray, shape(STATE_SIZE, INPUT_SIZE)
            differential of the model /alpha U

        Notes
        -------
        in this case, I pre-calculated the differential of the model because the tow-wheeled model is not difficult to calculate the gradient.
        If you dont or cannot do that, you can use the numerical differentiation.
        However, sometimes the the numerical differentiation affect the accuracy of calculations.
        """

        A = np.zeros((self.STATE_SIZE, self.STATE_SIZE))
        A_ideal = np.zeros((self.STATE_SIZE, self.STATE_SIZE))
        
        B = np.zeros((self.STATE_SIZE, self.INPUT_SIZE))
        B_ideal = np.zeros((self.STATE_SIZE, self.INPUT_SIZE))

        # if you want to use the numerical differentiation, please comment out this code
        """ 
        eps = 1e-4 # finite differences epsilon

        for ii in range(self.STATE_SIZE):
            # calculate partial differential w.r.t. x
            inc_x = x.copy()
            inc_x[ii] += eps
            state_inc,_ = self.plant_dynamics(inc_x, u.copy())
            dec_x = x.copy()
            dec_x[ii] -= eps
            state_dec,_ = self.plant_dynamics(dec_x, u.copy())
            A[:, ii] = (state_inc - state_dec) / (2 * eps)
        """

        A_ideal[0, 2] = -np.sin(x[2]) * u[0]
        A_ideal[1, 2] = np.cos(x[2]) * u[0]
        
        # if you want to use the numerical differentiation, please comment out this code
        """
        for ii in range(self.INPUT_SIZE):
            # calculate partial differential w.r.t. u
            inc_u = u.copy()
            inc_u[ii] += eps
            state_inc,_ = self.plant_dynamics(x.copy(), inc_u)
            dec_u = u.copy()
            dec_u[ii] -= eps
            state_dec,_ = self.plant_dynamics(x.copy(), dec_u)
            B[:, ii] = (state_inc - state_dec) / (2 * eps)
        """

        B_ideal[0, 0] = np.cos(x[2])
        B_ideal[1, 0] = np.sin(x[2])
        B_ideal[2, 1] = 1.

        return A_ideal, B_ideal

    def ilqr(self, x0, U=None): 
        """ use iterative linear quadratic regulation to find a control 
        sequence that minimizes the cost function 
        
        Parameters
        --------------
        x0 : numpy.ndarray, shape(STATE_SIZE, )
            the initial state of the system
        U : numpy.ndarray(TIME, INPUT_SIZE)
            the initial control trajectory dimension
        """
        U = self.U if U is None else U

        lamb = 1.0 # regularization parameter
        sim_new_trajectory = True
        tN = U.shape[0] # number of time steps

        for ii in range(self.max_iter):

            if sim_new_trajectory == True: 
                # simulate forward using the current control trajectory
                X, cost = self.simulate(x0, U)
                oldcost = np.copy(cost) # copy for exit condition check

                # 
                f_x = np.zeros((tN, self.STATE_SIZE, self.STATE_SIZE)) # df / dx
                f_u = np.zeros((tN, self.STATE_SIZE, self.INPUT_SIZE)) # df / du
                # for storing quadratized cost function 

                l = np.zeros((tN,1)) # immediate state cost 
                l_x = np.zeros((tN, self.STATE_SIZE)) # dl / dx
                l_xx = np.zeros((tN, self.STATE_SIZE, self.STATE_SIZE)) # d^2 l / dx^2
                l_u = np.zeros((tN, self.INPUT_SIZE)) # dl / du
                l_uu = np.zeros((tN, self.INPUT_SIZE, self.INPUT_SIZE)) # d^2 l / du^2
                l_ux = np.zeros((tN, self.INPUT_SIZE, self.STATE_SIZE)) # d^2 l / du / dx
                # for everything except final state
                for t in range(tN-1):
                    # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                    # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
                    # f_x = np.eye + A(t)
                    # f_u = B(t)
                    A, B = self.finite_differences(X[t], U[t])
                    f_x[t] = np.eye(self.STATE_SIZE) + A * self.dt
                    f_u[t] = B * self.dt
                
                    (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t]) = self.cost(X[t], U[t])
                    l[t] *= self.dt
                    l_x[t] *= self.dt
                    l_xx[t] *= self.dt
                    l_u[t] *= self.dt
                    l_uu[t] *= self.dt
                    l_ux[t] *= self.dt

                # and for final state
                l[-1], l_x[-1], l_xx[-1] = self.cost_final(X[-1])

                sim_new_trajectory = False

            V = l[-1].copy() # value function
            V_x = l_x[-1].copy() # dV / dx
            V_xx = l_xx[-1].copy() # d^2 V / dx^2
            k = np.zeros((tN, self.INPUT_SIZE)) # feedforward modification
            K = np.zeros((tN, self.INPUT_SIZE, self.STATE_SIZE)) # feedback gain

            # work backwards to solve for V, Q, k, and K
            for t in range(self.tN-2, -1, -1):
                Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
                Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

                Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
                Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
                Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

                Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += lamb
                Q_uu_inv = np.dot(Q_uu_evecs, np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

                k[t] = -np.dot(Q_uu_inv, Q_u)
                K[t] = -np.dot(Q_uu_inv, Q_ux)

                V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
                V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

            U_new = np.zeros((tN, self.INPUT_SIZE))
            x_new = x0.copy()
            for t in range(tN - 1): 
                # use feedforward (k) and feedback (K) gain matrices 
                # calculated from our value function approximation
                U_new[t] = U[t] + k[t] + np.dot(K[t], x_new - X[t])
                _,x_new = self.plant_dynamics(x_new, U_new[t])

            # evaluate the new trajectory 
            X_new, cost_new = self.simulate(x0, U_new)

            if cost_new < cost: 
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lamb_factor

                X = np.copy(X_new) # update trajectory 
                U = np.copy(U_new) # update control signal
                oldcost = np.copy(cost)
                cost = np.copy(cost_new)

                sim_new_trajectory = True # do another rollout

                # check to see if update is small enough to exit
                if ii > 0 and ((abs(oldcost-cost)/cost) < self.eps_converge):
                    print("Converged at iteration = %d; Cost = %.4f;"%(ii,cost_new) + 
                            " logLambda = %.1f"%np.log(lamb))
                    break

            else: 
                # increase lambda (get closer to gradient descent)
                lamb *= self.lamb_factor
                # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lamb_max: 
                    print("lambda > max_lambda at iteration = %d;"%ii + 
                        " Cost = %.4f; logLambda = %.1f"%(cost, 
                                                          np.log(lamb)))
                    break

        return X, U, cost

    def plant_dynamics(self, x, u):
        """ simulate a single time step of the plant, from 
        initial state x and applying control signal u

        Parameters
        --------------
        x : numpy.ndarray, shape(STATE_SIZE, )
            the state of the system
        u : numpy.ndarray, shape(INPUT_SIZE, )
            the control signal
        
        Returns
        -----------
        xdot : numpy.ndarray, shape(STATE_SIZE, )
            the gradient of x
        x_next : numpy.ndarray, shape(STATE_SIZE, )
            next state of x
        """ 
        self.simulator.initialize_state(x)

        # apply the control signal
        x_next = self.simulator.update_state(u, self.dt)
        
        # calculate the change in state
        xdot = ((x_next - x) / self.dt).squeeze()

        return xdot, x_next

    def reset(self, target):
        """ reset the state of the system """

        # Index along current control sequence
        self.t = 0
        self.U = np.zeros((self.tN, self.INPUT_SIZE))
        self.target = target.copy()

    def simulate(self, x0, U):
        """ do a rollout of the system, starting at x0 and 
        applying the control sequence U

        Parameters
        ----------
        x0 : numpy.ndarray, shape(STATE_SIZE, )
            the initial state of the system
        U : numpy.ndarray, shape(tN, INPUT_SIZE)
            the control sequence to apply
        
        Returns
        ---------
        X : numpy.ndarray, shape(tN, STATE_SIZE)
            the state sequence
        cost : float
            cost
        """ 
        
        tN = U.shape[0]
        X = np.zeros((tN, self.STATE_SIZE))
        X[0] = x0
        cost = 0

        # Run simulation with substeps
        for t in range(tN-1):
            _,X[t+1] = self.plant_dynamics(X[t], U[t])
            l, _ , _, _ , _ , _  = self.cost(X[t], U[t])
            cost = cost + self.dt * l

        # Adjust for final cost, subsample trajectory
        l_f, _, _ = self.cost_final(X[-1])
        cost = cost + l_f

        return X, cost
