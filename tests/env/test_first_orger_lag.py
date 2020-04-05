import pytest
import numpy as np

from PythonLinearNonlinearControl.envs.first_order_lag import FirstOrderLagEnv

class TestFirstOrderLagEnv():

    def test_step(self):
        env = FirstOrderLagEnv()

        curr_x = np.ones(4)

        env.reset(init_x=curr_x)
        
        u = np.ones(2) * 0.1

        next_x, _, _, _ = env.step(u)

        dx = np.dot(env.A, curr_x[:, np.newaxis])
        du = np.dot(env.B, u[:, np.newaxis])

        expected = (dx + du).flatten()

        assert next_x == pytest.approx(expected, abs=1e-5) 
    
    def test_bound_step(self):
        env = FirstOrderLagEnv()

        curr_x = np.ones(4)

        env.reset(init_x=curr_x)
        
        u = np.ones(2) * 1e5

        next_x, _, _, _ = env.step(u)

        dx = np.dot(env.A, curr_x[:, np.newaxis])
        du = np.dot(env.B,
                    np.array(env.config["input_upper_bound"])[:, np.newaxis])

        expected = (dx + du).flatten()

        assert next_x == pytest.approx(expected, abs=1e-5) 