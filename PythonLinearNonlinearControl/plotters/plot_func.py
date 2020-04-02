import os

import numpy as np
import matplotlib.pyplot as plt

def plot_result(history, history_g=None, ylabel="x",
                save_dir="./result", name="state_history"):
    """
    Args:
        history (numpy.ndarray): history, shape(iters, size)
    """
    (iters, size) = history.shape
    for i in range(0, size, 3):

        figure = plt.figure()
        axis1 = figure.add_subplot(311)
        axis2 = figure.add_subplot(312)
        axis3 = figure.add_subplot(313)

        axis1.set_ylabel(ylabel + "_{}".format(i))
        axis2.set_ylabel(ylabel + "_{}".format(i+1))
        axis3.set_ylabel(ylabel + "_{}".format(i+2))
        axis3.set_xlabel("time steps")

        # gt
        def plot(axis, history, history_g=None):
            axis.plot(range(iters), history, c="r", linewidth=3)
            if history_g is not None:
                axis.plot(range(iters), history_g,\
                          c="b", linewidth=3, label="goal")

        if i < size:
            plot(axis1, history[:, i], history_g=history_g[:, i])
        if i+1 < size:
            plot(axis2, history[:, i+1], history_g=history_g[:, i+1])
        if i+2 < size:    
            plot(axis3, history[:, i+2], history_g=history_g[:, i+2])

        # save
        if save_dir is not None:
            path = os.path.join(save_dir, name + "-{}".format(i))
        else:
            path = name

        axis1.legend(ncol=1, bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3)
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05)

def plot_results(args, history_x, history_u, history_g=None):
    """
    Args:
        history_x (numpy.ndarray): history of state, shape(iters, state_size)
        history_u (numpy.ndarray): history of state, shape(iters, input_size)
    Returns:
    """
    plot_result(history_x, history_g=history_g, ylabel="x",
                name="state_history",
                save_dir="./result/" + args.controller_type)
    plot_result(history_u, history_g=np.zeros_like(history_u), ylabel="u",
                name="input_history",
                save_dir="./result/" + args.controller_type)