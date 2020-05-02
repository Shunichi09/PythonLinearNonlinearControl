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