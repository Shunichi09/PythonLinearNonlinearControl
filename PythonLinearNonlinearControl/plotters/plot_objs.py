import os

import numpy as np
import matplotlib.pyplot as plt

from ..common.utils import rotate_pos


def circle(center_x, center_y, radius, start=0., end=2*np.pi, n_point=100):
    """ Create circle matrix

    Args:
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        start (float): start angle
        end (float): end angle
    Returns:
        circle x : numpy.ndarray
        circle y : numpy.ndarray
    """
    diff = end - start

    circle_xs = []
    circle_ys = []

    for i in range(n_point + 1):
        circle_xs.append(center_x + radius * np.cos(i*diff/n_point + start))
        circle_ys.append(center_y + radius * np.sin(i*diff/n_point + start))

    return np.array(circle_xs), np.array(circle_ys)


def circle_with_angle(center_x, center_y, radius, angle):
    """ Create circle matrix with angle line matrix

    Args:    
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        angle (float): in radians
    Returns: 
        circle_x (numpy.ndarray): x data of circle 
        circle_y (numpy.ndarray): y data of circle
        angle_x (numpy.ndarray): x data of circle angle
        angle_y (numpy.ndarray): y data of circle angle
    """
    circle_x, circle_y = circle(center_x, center_y, radius)

    angle_x = np.array([center_x, center_x + np.cos(angle) * radius])
    angle_y = np.array([center_y, center_y + np.sin(angle) * radius])

    return circle_x, circle_y, angle_x, angle_y


def square(center_x, center_y, shape, angle):
    """ Create square

    Args:    
        center_x (float): the center x position of the square
        center_y (float): the center y position of the square
        shape (tuple): the square's shape(width/2, height/2)
        angle (float): in radians
    Returns: 
        square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
        square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
    """
    # start with the up right points
    # create point in counterclockwise, local
    square_xy = np.array([[shape[0], shape[1]],
                          [-shape[0], shape[1]],
                          [-shape[0], -shape[1]],
                          [shape[0], -shape[1]],
                          [shape[0], shape[1]]])
    # translate position to world
    # rotation
    trans_points = rotate_pos(square_xy, angle)
    # translation
    trans_points += np.array([center_x, center_y])

    return trans_points[:, 0], trans_points[:, 1]


def square_with_angle(center_x, center_y, shape, angle):
    """ Create square with angle line

    Args:    
        center_x (float): the center x position of the square
        center_y (float): the center y position of the square
        shape (tuple): the square's shape(width/2, height/2)
        angle (float): in radians
    Returns: 
        square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
        square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
        angle_x (numpy.ndarray): x data of square angle
        angle_y (numpy.ndarray): y data of square angle
    """
    square_x, square_y = square(center_x, center_y, shape, angle)

    angle_x = np.array([center_x, center_x + np.cos(angle) * shape[0]])
    angle_y = np.array([center_y, center_y + np.sin(angle) * shape[1]])

    return square_x, square_y, angle_x, angle_y
