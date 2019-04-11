import numpy as np
import matplotlib.pyplot as plt
import math

from model import TwoWheeledCar
from ilqr import iLQRController
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
    goal_maker = 

    # controller
    controller = iLQRController()


    for iteration in range(NUM_ITERATIONS):
        print("iteration num = {} / {}".format(iteration, NUM_ITERATIONS))

        u = controller.calc_input(car, target)
        car.update_state(u, dt) # update state

    # figures and animation
    history_states = np.array(car.history_xs)

    time_fig = plt.figure(figsize=(3, 4))

    x_fig = time_fig.add_subplot(311)
    y_fig = time_fig.add_subplot(312)
    th_fig = time_fig.add_subplot(313)

    time = len(history_states)
    x_fig.plot(np.arange(time), history_states[:, 0])
    y_fig.plot(np.arange(time), history_states[:, 1])
    th_fig.plot(np.arange(time), history_states[:, 2])

    plt.show()

    history_states = np.array(car.history_xs)

    animdrawer = AnimDrawer([history_states, target])
    animdrawer.draw_anim()

if __name__ == "__main__":
    main()