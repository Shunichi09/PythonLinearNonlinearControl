import numpy as np


def rotate_pos(pos, angle):
    """ Transformation the coordinate in the angle

    Args:
        pos (numpy.ndarray): local state, shape(data_size, 2) 
        angle (float): rotate angle, in radians
    Returns:
        rotated_pos (numpy.ndarray): shape(data_size, 2)
    """
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

    return np.dot(pos, rot_mat.T)


def fit_angle_in_range(angles, min_angle=-np.pi, max_angle=np.pi):
    """ Check angle range and correct the range

    Args:
        angle (numpy.ndarray): in radians
        min_angle (float): maximum of range in radians, default -pi
        max_angle (float): minimum of range in radians, default pi
    Returns: 
        fitted_angle (numpy.ndarray): range angle in radians
    """
    if max_angle < min_angle:
        raise ValueError("max angle must be greater than min angle")
    if (max_angle - min_angle) < 2.0 * np.pi:
        raise ValueError("difference between max_angle \
                          and min_angle must be greater than 2.0 * pi")

    output = np.array(angles)
    output_shape = output.shape

    output = output.flatten()
    output -= min_angle
    output %= 2 * np.pi
    output += 2 * np.pi
    output %= 2 * np.pi
    output += min_angle

    output = np.minimum(max_angle, np.maximum(min_angle, output))
    return output.reshape(output_shape)


def update_state_with_Runge_Kutta(state, u, functions, dt=0.01):
    """ update state in Runge Kutta methods
    Args:
        state (array-like): state of system
        u (array-like): input of system
        functions (list): update function of each state,
            each function will be called like func(*state, *u)
            We expect that this function returns differential of each state 
        dt (float): float in seconds

    Returns:
        next_state (np.array): next state of system

    Notes:
        sample of function is as follows:

        def func_x(self, x_1, x_2, u):
            x_dot = (1. - x_1**2 - x_2**2) * x_2 - x_1 + u
            return x_dot

        Note that the function return x_dot.
    """
    state_size = len(state)
    assert state_size == len(functions), \
        "Invalid functions length, You need to give the state size functions"

    k0 = np.zeros(state_size)
    k1 = np.zeros(state_size)
    k2 = np.zeros(state_size)
    k3 = np.zeros(state_size)

    inputs = np.concatenate([state, u])

    for i, func in enumerate(functions):
        k0[i] = dt * func(*inputs)

    add_state = state + k0 / 2.
    inputs = np.concatenate([add_state, u])

    for i, func in enumerate(functions):
        k1[i] = dt * func(*inputs)

    add_state = state + k1 / 2.
    inputs = np.concatenate([add_state, u])

    for i, func in enumerate(functions):
        k2[i] = dt * func(*inputs)

    add_state = state + k2
    inputs = np.concatenate([add_state, u])

    for i, func in enumerate(functions):
        k3[i] = dt * func(*inputs)

    return (k0 + 2. * k1 + 2. * k2 + k3) / 6.
