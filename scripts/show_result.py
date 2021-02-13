import os

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from PythonLinearNonlinearControl.plotters.plot_func import load_plot_data, \
    plot_multi_result


def run(args):

    controllers = ["iLQR", "DDP", "CEM", "MPPI"]

    history_xs = None
    history_us = None
    history_gs = None

    # load data
    for controller in controllers:
        history_x, history_u, history_g = \
            load_plot_data(args.env, controller,
                           result_dir=args.result_dir)

        if history_xs is None:
            history_xs = history_x[np.newaxis, :]
            history_us = history_u[np.newaxis, :]
            history_gs = history_g[np.newaxis, :]
            continue

        history_xs = np.concatenate((history_xs,
                                     history_x[np.newaxis, :]), axis=0)
        history_us = np.concatenate((history_us,
                                     history_u[np.newaxis, :]), axis=0)
        history_gs = np.concatenate((history_gs,
                                     history_g[np.newaxis, :]), axis=0)

    plot_multi_result(history_xs, histories_g=history_gs, labels=controllers,
                      ylabel="x")

    plot_multi_result(history_us, histories_g=np.zeros_like(history_us),
                      labels=controllers, ylabel="u", name="input_history")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="FirstOrderLag")
    parser.add_argument("--result_dir", type=str, default="./result")

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
