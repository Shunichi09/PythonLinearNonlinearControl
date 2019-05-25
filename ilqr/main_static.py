import numpy as np
import matplotlib.pyplot as plt
import math

from model import TwoWheeledCar
from ilqr import iLQRController
from goal_maker import GoalMaker
from animation import AnimDrawer


def main():
    """
    """
    # iteration parameters
    NUM_ITERATIONS = 500
    dt = 0.01

    # make plant
    init_x = np.array([0., 0., 0.5*math.pi])
    car = TwoWheeledCar(init_x)

    # make goal
    goal_maker = GoalMaker(goal_type="const")
    goal_maker.preprocess()

    # controller
    controller = iLQRController()

    for iteration in range(NUM_ITERATIONS):
        print("iteration num = {} / {}".format(iteration, NUM_ITERATIONS))

        target = goal_maker.calc_goal(car.xs)
        u = controller.calc_input(car, target)
        car.update_state(u, dt) # update state

    # figures and animation
    history_states = np.array(car.history_xs)
    history_targets = np.array(goal_maker.history_target)

    time_fig = plt.figure()

    x_fig = time_fig.add_subplot(311)
    y_fig = time_fig.add_subplot(312)
    th_fig = time_fig.add_subplot(313)

    time = len(history_states)

    x_fig.plot(np.arange(time), history_states[:, 0], "r")
    x_fig.plot(np.arange(1, time), history_targets[:, 0], "b", linestyle="dashdot")
    x_fig.set_ylabel("x")

    y_fig.plot(np.arange(time), history_states[:, 1], "r")
    y_fig.plot(np.arange(1, time), history_targets[:, 1], "b", linestyle="dashdot")
    y_fig.set_ylabel("y")

    th_fig.plot(np.arange(time), history_states[:, 2], "r")
    th_fig.plot(np.arange(1, time), history_targets[:, 2], "b", linestyle="dashdot")
    th_fig.set_ylabel("th")

    plt.show()

    history_states = np.array(car.history_xs)

    animdrawer = AnimDrawer([history_states, history_targets])
    animdrawer.draw_anim()

if __name__ == "__main__":
    main()