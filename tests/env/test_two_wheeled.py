import pytest
import numpy as np

from PythonLinearNonlinearControl.envs.two_wheeled import TwoWheeledConstEnv
    
class TestTwoWheeledEnv():
    """
    """
    def test_step(self):
        env = TwoWheeledConstEnv()

        curr_x = np.ones(3)
        curr_x[-1] = np.pi / 6.

        env.reset(init_x=curr_x)
        
        u = np.ones(2)

        next_x, _, _, _ = env.step(u)

        pos_x = np.cos(curr_x[-1]) * u[0] * env.config["dt"] + curr_x[0]
        pos_y = np.sin(curr_x[-1]) * u[0] * env.config["dt"] + curr_x[1]

        expected = np.array([pos_x, pos_y,\
                             curr_x[-1] + u[1] * env.config["dt"]])

        assert next_x == pytest.approx(expected) 
    
    def test_bound_step(self):
        env = TwoWheeledConstEnv()

        curr_x = np.ones(3)
        curr_x[-1] = np.pi / 6.

        env.reset(init_x=curr_x)

        u = np.ones(2) * 1e3

        next_x, _, _, _ = env.step(u)

        pos_x = np.cos(curr_x[-1]) * env.config["input_upper_bound"][0] \
                * env.config["dt"] + curr_x[0]
        pos_y = np.sin(curr_x[-1]) * env.config["input_upper_bound"][0] \
                * env.config["dt"] + curr_x[1]

        expected = np.array([pos_x, pos_y,\
                             curr_x[-1] + env.config["input_upper_bound"][1] \
                             * env.config["dt"]])

        assert next_x == pytest.approx(expected)