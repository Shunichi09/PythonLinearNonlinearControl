import os

import numpy as np
import matplotlib.pyplot as plt

from ..helper import save_pickle, load_pickle


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
                axis.plot(range(iters), history_g,
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


def plot_results(history_x, history_u, history_g=None, args=None):
    """

    Args:
        history_x (numpy.ndarray): history of state, shape(iters, state_size)
        history_u (numpy.ndarray): history of state, shape(iters, input_size)
    Returns:
        None
    """
    env = "Env"
    controller_type = "controller"

    if args is not None:
        env = args.env
        controller_type = args.controller_type

    plot_result(history_x, history_g=history_g, ylabel="x",
                name=env + "-state_history",
                save_dir="./result/" + controller_type)
    plot_result(history_u, history_g=np.zeros_like(history_u), ylabel="u",
                name=env + "-input_history",
                save_dir="./result/" + controller_type)


def save_plot_data(history_x, history_u, history_g=None, args=None):
    """ save plot data

    Args:
        history_x (numpy.ndarray): history of state, shape(iters, state_size)
        history_u (numpy.ndarray): history of state, shape(iters, input_size)
    Returns:
        None
    """
    env = "Env"
    controller_type = "controller"

    if args is not None:
        env = args.env
        controller_type = args.controller_type

    path = os.path.join("./result/" + controller_type,
                        env + "-history_x.pkl")
    save_pickle(path, history_x)

    path = os.path.join("./result/" + controller_type,
                        env + "-history_u.pkl")
    save_pickle(path, history_u)

    path = os.path.join("./result/" + controller_type,
                        env + "-history_g.pkl")
    save_pickle(path, history_g)


def load_plot_data(env, controller_type, result_dir="./result"):
    """
    Args:
        env (str): environments name
        controller_type (str): controller type
        result_dir (str): result directory
    Returns:
        history_x (numpy.ndarray): history of state, shape(iters, state_size)
        history_u (numpy.ndarray): history of state, shape(iters, input_size)
        history_g (numpy.ndarray): history of state, shape(iters, input_size)
    """
    path = os.path.join("./result/" + controller_type,
                        env + "-history_x.pkl")
    history_x = load_pickle(path)

    path = os.path.join("./result/" + controller_type,
                        env + "-history_u.pkl")
    history_u = load_pickle(path)

    path = os.path.join("./result/" + controller_type,
                        env + "-history_g.pkl")
    history_g = load_pickle(path)

    return history_x, history_u, history_g


def plot_multi_result(histories, histories_g=None, labels=None, ylabel="x",
                      save_dir="./result", name="state_history"):
    """
    Args:
        history (numpy.ndarray): history, shape(iters, size)
    """
    (_, iters, size) = histories.shape

    for i in range(0, size, 2):

        figure = plt.figure()
        axis1 = figure.add_subplot(211)
        axis2 = figure.add_subplot(212)

        axis1.set_ylabel(ylabel + "_{}".format(i))
        axis2.set_ylabel(ylabel + "_{}".format(i+1))
        axis2.set_xlabel("time steps")

        # gt
        def plot(axis, history, history_g=None, label=""):
            axis.plot(range(iters), history,
                      linewidth=3, label=label, alpha=0.7, linestyle="dashed")
            if history_g is not None:
                axis.plot(range(iters), history_g,
                          c="b", linewidth=3)

        if i < size:
            for j, (history, history_g) \
                    in enumerate(zip(histories, histories_g)):
                plot(axis1, history[:, i],
                     history_g=history_g[:, i], label=labels[j])
        if i+1 < size:
            for j, (history, history_g) in \
                    enumerate(zip(histories, histories_g)):
                plot(axis2, history[:, i+1],
                     history_g=history_g[:, i+1], label=labels[j])

        # save
        if save_dir is not None:
            path = os.path.join(save_dir, name + "-{}".format(i))
        else:
            path = name

        axis1.legend(ncol=3, bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3)
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05)
