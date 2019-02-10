import numpy as np
import matplotlib.pyplot as plt
import math

import random

from traj_func import make_sample_traj

def calc_curvature(points):
    """
    Parameters
    -----------
    points : numpy.ndarray, shape (2, 3)
        these points should follow subseqently
    
    Returns
    ----------
    curvature : float
    """
    # Gradient 1
    diff = points[:, 0] - points[:, 1]
    Gradient_1 = -1. / (diff[1] / diff[0])
    # Gradient 2
    diff = points[:, 1] - points[:, 2]
    Gradient_2 = -1. / (diff[1] / diff[0])

    # middle point 1
    middle_point_1 = (points[:, 0] + points[:, 1]) / 2.

    # middle point 2
    middle_point_2 = (points[:, 1] + points[:, 2]) / 2.

    # calc center
    c_x = (middle_point_1[1] - middle_point_2[1] - middle_point_1[0] * Gradient_1 + middle_point_2[0] * Gradient_2) / (Gradient_2 - Gradient_1)
    c_y = middle_point_1[1] - (middle_point_1[0] - c_x) * Gradient_1

    R = math.sqrt((points[0, 0] - c_x)**2 + (points[1, 0] - c_y)**2)

    """
    plt.scatter(points[0, :], points[1, :])
    plt.scatter(c_x, c_y)

    plot_points_x = []
    plot_points_y = []

    for theta in np.arange(0, 2*math.pi, 0.01):
        plot_points_x.append(math.cos(theta)*R + c_x)
        plot_points_y.append(math.sin(theta)*R + c_y)

    plt.plot(plot_points_x, plot_points_y)

    plt.show()
    """

    return 1. / R

def calc_curvatures(traj_ref, predict_step, num_skip):
    """
    Parameters
    -----------
    traj_ref : numpy.ndarray, shape (2, N)
        these points should follow subseqently
    predict_step : int
        predict step
    num_skip : int
        skip_num 
    
    Returns
    ----------
    angles : list
    curvature : list
    """

    angles = []
    curvatures = []

    for i in range(predict_step):
        # make pairs
        points = np.zeros((2, 3))

        points[:, 0] = traj_ref[:, i]
        points[:, 1] = traj_ref[:, i + num_skip]
        points[:, 2] = traj_ref[:, i + 2 * num_skip]

        # Gradient 1
        diff = points[:, 0] - points[:, 1]
        Gradient_1 = -1. / (diff[1] / diff[0])
        # Gradient 2
        diff = points[:, 1] - points[:, 2]
        Gradient_2 = -1. / (diff[1] / diff[0])

        # middle point 1
        middle_point_1 = (points[:, 0] + points[:, 1]) / 2.

        # middle point 2
        middle_point_2 = (points[:, 1] + points[:, 2]) / 2.

        # calc center
        c_x = (middle_point_1[1] - middle_point_2[1] - middle_point_1[0] * Gradient_1 + middle_point_2[0] * Gradient_2) / (Gradient_2 - Gradient_1)
        c_y = middle_point_1[1] - (middle_point_1[0] - c_x) * Gradient_1

        # calc R
        R = math.sqrt((points[0, 0] - c_x)**2 + (points[1, 0] - c_y)**2)

        # add
        diff = points[:, 2] - points[:, 0]
        angles.append(math.atan2(diff[1], diff[0]))
        curvatures.append(1. / R)

        # plot
        """
        plt.scatter(points[0, :], points[1, :])
        plt.scatter(c_x, c_y)

        plot_points_x = []
        plot_points_y = []

        for theta in np.arange(0, 2*math.pi, 0.01):
            plot_points_x.append(math.cos(theta)*R + c_x)
            plot_points_y.append(math.sin(theta)*R + c_y)

        plt.plot(plot_points_x, plot_points_y)

        plot_points_x = []
        plot_points_y = []

        for x in np.arange(-5, 5, 0.01):
            plot_points_x.append(x)
            plot_points_y.append(x * math.tan(angles[-1]))

        plt.plot(plot_points_x, plot_points_y)

        plt.xlim(-1, 10)
        plt.ylim(-1, 2)

        plt.show()
        """

    return angles, curvatures

def calc_ideal_vel(traj_ref, dt):
    """
    Parameters
    ------------
    traj_ref : numpy.ndarray, shape (2, N)
        these points should follow subseqently
    dt : float
        sampling time of system
    """
    # end point and start point
    diff = traj_ref[:, -1] - traj_ref[:, 0] 
    distance = np.sqrt(np.sum(diff**2))

    V = distance / (dt * traj_ref.shape[1])

    return V

def main():
    """
    points = np.zeros((2, 3))
    points[:, 0] = np.array([1. +  random.random(), random.random()])

    points[:, 1] = np.array([random.random(), 3 + random.random()])

    points[:, 2] = np.array([3 + random.random(), -3 + random.random()])

    calc_cuvature(points)
    """

    traj_ref_xs, traj_ref_ys = make_sample_traj(1000)
    traj_ref = np.array([traj_ref_xs, traj_ref_ys])

    calc_curvatures(traj_ref[:, 10:10 + 15 + 100 * 2], 15, 100)


if __name__ == "__main__":
    main()





