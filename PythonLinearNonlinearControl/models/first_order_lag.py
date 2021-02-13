import numpy as np
import scipy.linalg
from scipy import integrate
from .model import LinearModel


class FirstOrderLagModel(LinearModel):
    """ first order lag model
    Attributes:
        curr_x (numpy.ndarray):
        u (numpy.ndarray):
        history_pred_xs (numpy.ndarray):
    """

    def __init__(self, config, tau=0.63):
        """
        Args:
            tau (float): time constant 
        """
        # param
        self.A, self.B = self._to_state_space(
            tau, dt=config.DT)  # discrete system
        super(FirstOrderLagModel, self).__init__(self.A, self.B)

    @staticmethod
    def _to_state_space(tau, dt=0.05):
        """
        Args:
            tau (float): time constant
            dt (float): discrte time
        Returns:
            A (numpy.ndarray): discrete A matrix 
            B (numpy.ndarray): discrete B matrix 
        """
        # continuous
        Ac = np.array([[-1./tau, 0., 0., 0.],
                       [0., -1./tau, 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.]])
        Bc = np.array([[1./tau, 0.],
                       [0., 1./tau],
                       [0., 0.],
                       [0., 0.]])
        # to discrete system
        A = scipy.linalg.expm(dt*Ac)
        # B = np.matmul(np.matmul(scipy.linalg.expm(Ac*dt)-scipy.linalg.expm(Ac*0.), np.linalg.inv(Ac)), Bc)
        B = np.zeros_like(Bc)
        for m in range(Bc.shape[0]):
            for n in range(Bc.shape[1]):
                def integrate_fn(tau): return np.matmul(
                    scipy.linalg.expm(Ac*tau), Bc)[m, n]
                sol = integrate.quad(integrate_fn, 0, dt)
                B[m, n] = sol[0]

        return A, B
