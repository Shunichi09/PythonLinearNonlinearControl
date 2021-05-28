import os
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

logger = getLogger(__name__)


class Animator():
    """ animation class
    """

    def __init__(self, env, args=None):
        """
        """
        self.env_name = "Env"
        self.result_dir = "./result"
        self.controller_type = "controller"

        if args is not None:
            self.env_name = args.env
            self.result_dir = args.result_dir
            self.controller_type = args.controller_type

        self.interval = env.config["dt"] * 1000.  # to ms
        self.plot_func = env.plot_func

        self.imgs = []

    def _setup(self):
        """ set up figure of animation
        """
        # make fig
        self.anim_fig = plt.figure()

        # axis
        self.axis = self.anim_fig.add_subplot(111)
        self.axis.set_aspect('equal', adjustable='box')

        self.imgs = self.plot_func(self.axis)

    def _update_img(self, i, history_x, history_g_x):
        """ update animation

        Args:
            i (int): frame count
            history_x (numpy.ndarray): history of state,
                                       shape(iters, state_size)
            history_g (numpy.ndarray): history of goal state,
                                       shape(iters, input_size)
        """
        self.plot_func(self.imgs, i, history_x, history_g_x)

    def draw(self, history_x, history_g_x=None):
        """draw the animation and save

        Args:
            history_x (numpy.ndarray): history of state,
                                       shape(iters, state_size)
            history_g (numpy.ndarray): history of goal state,
                                       shape(iters, input_size)
        Returns:
            None
        """
        # set up animation figures
        self._setup()
        def _update_img(i): return self._update_img(i, history_x, history_g_x)

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        # call funcanimation
        ani = FuncAnimation(
            self.anim_fig,
            _update_img, interval=self.interval, frames=len(history_x)-1)

        # save animation
        path = os.path.join(self.result_dir, self.controller_type,
                            "animation-" + self.env_name + ".mp4")
        logger.info("Saved Animation to {} ...".format(path))

        ani.save(path, writer=writer)
