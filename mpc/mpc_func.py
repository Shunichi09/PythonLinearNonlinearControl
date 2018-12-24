import numpy as np
import matplotlib.pyplot as plt
import math
import copy

from scipy.optimize import minimize

class MpcController():
    """
    Attributes
    ------------

    """

    def __init__(self, A, B, Q, R, pre_step, input_upper=None, input_lower=None):
        """
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.pre_step = pre_step

        self.Qs = None
        self.Rs = None

        self.state_size = self.A.shape[0]
        self.input_size = self.B.shape[1]

        self.history_us = []

    def initialize_controller(self):
        """
        make matrix to calculate optimal controller

        """
        A_factorials = [self.A]

        self.phi_mat = copy.deepcopy(self.A)

        for _ in range(self.pre_step - 1):
            temp_mat = np.dot(A_factorials[-1], self.A)
            self.phi_mat = np.vstack((self.phi_mat, temp_mat))

            A_factorials.append(temp_mat) # after we use this factorials
            
        print("phi_mat = \n{0}".format(self.phi_mat))

        self.gamma_mat = copy.deepcopy(self.B)
        gammma_mat_temp = copy.deepcopy(self.B)
        
        for i in range(self.pre_step - 1):
            temp_1_mat = np.dot(A_factorials[i], self.B)
            gammma_mat_temp = temp_1_mat + gammma_mat_temp
            self.gamma_mat = np.vstack((self.gamma_mat, gammma_mat_temp))

        print("gamma_mat = \n{0}".format(self.gamma_mat))

        self.theta_mat = copy.deepcopy(self.gamma_mat)

        for i in range(self.pre_step - 1):
            temp_mat = np.zeros_like(self.gamma_mat)
            temp_mat[int((i + 1)*self.state_size): , :] = self.gamma_mat[:-int((i + 1)*self.state_size) , :]

            self.theta_mat = np.hstack((self.theta_mat, temp_mat))

        print("theta_mat = \n{0}".format(self.theta_mat))

        diag_Qs = np.array([np.diag(self.Q) for _ in range(self.pre_step)])
        diag_Rs = np.array([np.diag(self.R) for _ in range(self.pre_step)])
        
        self.Qs = np.diag(diag_Qs.flatten())
        self.Rs = np.diag(diag_Rs.flatten())

        print("Qs = {0}".format(self.Qs))
        print("Rs = {0}".format(self.Rs))

    def calc_input(self, states, references):
        """
        Parameters
        -----------
        states : numpy.array
            the size should have (state length * 1)
        references :
            the size should have (state length * pre_step)

        """
        temp_1 = np.dot(self.phi_mat, states)
        temp_2 = np.dot(self.gamma_mat, self.history_us[-1])

        error = references - temp_1 - temp_2

        G = 2. * np.dot(self.theta_mat.T, np.dot(self.Qs, error) )

        H = np.dot(self.theta_mat.T, np.dot(self.Qs, self.theta_mat)) + self.Rs

        def optimized_func(dt_us):
            """
            """
            return np.dot(dt_us.T, np.dot(H, dt_us)) - np.dot(G.T, dt_us)

        init_dt_us = np.zeros(self.pre_step)

        opt_result = minimize(optimized_func, init_dt_us)

        opt_dt_us = opt_result

        opt_us = opt_dt_us[0] + self.history_us[-1]

        # save
        self.history_us.append(opt_us)
        return opt_us


    





        

        



