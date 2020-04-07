import pytest
import numpy as np

from PythonLinearNonlinearControl.envs.cartpole import CartPoleEnv
    
class TestCartPoleEnv():
    """
    """
    def test_step(self):
        env = CartPoleEnv()

        curr_x = np.ones(4)
        curr_x[2] = np.pi / 6.

        env.reset(init_x=curr_x)
        
        u = np.ones(1)

        next_x, _, _, _ = env.step(u)

        d_x0 = curr_x[1]
        d_x1 = (1. + env.config["mp"] * np.sin(np.pi / 6.) \
                     * (env.config["l"] * (1.**2) \
                        + env.config["g"] * np.cos(np.pi / 6.))) \
                / (env.config["mc"] + env.config["mp"] * np.sin(np.pi / 6.)**2)
        d_x2 = curr_x[3]
        d_x3 = (-1. * np.cos(np.pi / 6.) \
                - env.config["mp"] * env.config["l"] * (1.**2) \
                  * np.cos(np.pi / 6.) * np.sin(np.pi / 6.) \
                - (env.config["mp"] + env.config["mc"]) * env.config["g"] \
                   * np.sin(np.pi / 6.)) \
                 / (env.config["l"] \
                     * (env.config["mc"] \
                        + env.config["mp"] * np.sin(np.pi / 6.)**2))

        expected = np.array([d_x0, d_x1, d_x2, d_x3]) * env.config["dt"] \
                   + curr_x

        assert next_x == pytest.approx(expected, abs=1e-5) 
    
    def test_bound_step(self):
        env = CartPoleEnv()

        curr_x = np.ones(4)
        curr_x[2] = np.pi / 6.

        env.reset(init_x=curr_x)
        
        u = np.ones(1) * 1e3

        next_x, _, _, _ = env.step(u)

        u = env.config["input_upper_bound"][0]

        d_x0 = curr_x[1]
        d_x1 = (u + env.config["mp"] * np.sin(np.pi / 6.) \
                     * (env.config["l"] * (1.**2) \
                        + env.config["g"] * np.cos(np.pi / 6.))) \
                / (env.config["mc"] + env.config["mp"] * np.sin(np.pi / 6.)**2)
        d_x2 = curr_x[3]
        d_x3 = (-u * np.cos(np.pi / 6.) \
                - env.config["mp"] * env.config["l"] * (1.**2) \
                  * np.cos(np.pi / 6.) * np.sin(np.pi / 6.) \
                - (env.config["mp"] + env.config["mc"]) * env.config["g"] \
                   * np.sin(np.pi / 6.)) \
                 / (env.config["l"] \
                     * (env.config["mc"] \
                        + env.config["mp"] * np.sin(np.pi / 6.)**2))

        expected = np.array([d_x0, d_x1, d_x2, d_x3]) * env.config["dt"] \
                   + curr_x

        assert next_x == pytest.approx(expected, abs=1e-5) 