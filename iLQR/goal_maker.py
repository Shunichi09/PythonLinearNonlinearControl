import numpy as np
import math
import matplotlib.pyplot as plt

def make_trajectory(goal_type, N):
    """
    Parameters
    -------------
    goal_type : str
        goal type
    N : int
        length of reference trajectory
    Returns
    -----------
    ref_trajectory : numpy.ndarray, shape(N, STATE_SIZE)

    Notes
    ---------
    this function can only deal with the 
    """
    
    if goal_type == "const":
        ref_trajectory = np.array([[5., 3., 0.]])

        return ref_trajectory
    
    if goal_type == "sin":
        # parameters
        amplitude = 2.
        num_period = 2.

        ref_x_trajectory = np.linspace(0., 2 * math.pi * num_period, N)
        ref_y_trajectory = amplitude * np.sin(ref_x_trajectory)
        
        ref_xy_trajectory = np.stack((ref_x_trajectory, ref_y_trajectory))

        # target of theta is always zero
        ref_trajectory = np.vstack((ref_xy_trajectory, np.zeros((1, N))))
        
        return ref_trajectory.T

class GoalMaker():
    """
    Attributes
    -----------
    N : int
        length of reference
    goal_type : str
        goal type
    dt : float
        sampling time
    ref_traj : numpy.ndarray, shape(N, STATE_SIZE)
        in this case the state size is 3
    """

    def __init__(self, N=500, goal_type="const", dt=0.01):
        """
        Parameters
        --------------
        N : int
            length of reference
        goal_type : str
            goal type
        dt : float
            sampling time
        """
        self.N = N
        self.goal_type = goal_type
        self.dt = dt

        self.ref_traj = None
        self.history_target = []
        
    def preprocess(self):
        """preprocess of make goal
        """
        goal_type_list = ["const", "sin"]

        if self.goal_type not in goal_type_list:
            raise KeyError("{0} is not in implemented goal type. please implement that!!".format(self.goal_type))

        self.ref_traj = make_trajectory(self.goal_type, self.N)

    def calc_goal(self, x):
        """ calclate nearest goal
        Parameters
        ------------
        x : numpy.ndarray, shape(STATE_SIZE, )
            state of the system
        
        Returns
        ------------
        goal : numpy.ndarray, shape(STATE_SIZE, )
        """

        # search nearest point
        x_dis = (self.ref_traj[:, 0]-x[0])**2
        y_dis = (self.ref_traj[:, 1]-x[1])**2
        index = np.argmin(np.sqrt(x_dis + y_dis))
        
        print(index)

        MARGIN = 30
        if not self.goal_type == "const":
            index += MARGIN
        
        if index > self.N-1:
            index = self.N - 1

        goal = self.ref_traj[index]

        self.history_target.append(goal)

        return goal

if __name__ == "__main__":
    make_trajectory("sin", 400)