import pytest
import numpy as np

from PythonLinearNonlinearControl.configs.cartpole \
    import CartPoleConfigModule

class TestCalcCost():
    def test_calc_costs(self):
        # make config
        config = CartPoleConfigModule()
        # set
        pred_len = 5
        state_size = 4
        input_size = 1
        pop_size = 2
        pred_xs = np.ones((pop_size, pred_len, state_size))
        g_xs = np.ones((pop_size, pred_len, state_size)) * 0.5
        input_samples = np.ones((pop_size, pred_len, input_size)) * 0.5

        costs = config.input_cost_fn(input_samples)

        assert costs.shape == (pop_size, pred_len, input_size)

        costs = config.state_cost_fn(pred_xs, g_xs)
        
        assert costs.shape == (pop_size, pred_len, 1)

        costs = config.terminal_state_cost_fn(pred_xs[:, -1, :],\
                                              g_xs[:, -1, :])
        
        assert costs.shape == (pop_size, 1)