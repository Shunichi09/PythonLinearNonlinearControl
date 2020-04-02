import pytest
import numpy as np

from PythonLinearNonlinearControl.configs.first_order_lag \
    import FirstOrderLagConfigModule

class TestCalcCost():
    def test_calc_costs(self):
        # make config
        config = FirstOrderLagConfigModule()
        # set
        pred_len = 5
        state_size = 4
        input_size = 2
        pop_size = 2
        pred_xs = np.ones((pop_size, pred_len, state_size))
        g_xs = np.ones((pop_size, pred_len, state_size)) * 0.5
        input_samples = np.ones((pop_size, pred_len, input_size)) * 0.5

        costs = config.input_cost_fn(input_samples)
        expected_costs = np.ones((pop_size, pred_len, input_size))*0.5

        assert costs == pytest.approx(expected_costs**2 * np.diag(config.R))

        costs = config.state_cost_fn(pred_xs, g_xs)
        expected_costs = np.ones((pop_size, pred_len, state_size))*0.5
        
        assert costs == pytest.approx(expected_costs**2 * np.diag(config.Q))

        costs = config.terminal_state_cost_fn(pred_xs[:, -1, :],\
                                              g_xs[:, -1, :])
        expected_costs = np.ones((pop_size, state_size))*0.5
        
        assert costs == pytest.approx(expected_costs**2 * np.diag(config.Sf))