import pytest
import numpy as np

from PythonLinearNonlinearControl.configs.two_wheeled \
    import TwoWheeledConfigModule


class TestCalcCost():
    def test_calc_costs(self):
        # make config
        config = TwoWheeledConfigModule()
        # set
        pred_len = 5
        state_size = 3
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

        costs = config.terminal_state_cost_fn(pred_xs[:, -1, :],
                                              g_xs[:, -1, :])
        expected_costs = np.ones((pop_size, state_size))*0.5

        assert costs == pytest.approx(expected_costs**2 * np.diag(config.Sf))


class TestGradient():
    def test_state_gradient(self):
        """
        """
        xs = np.ones((1, 3))
        g_xs = np.ones((1, 3)) * 0.5
        cost_grad =\
            TwoWheeledConfigModule.gradient_cost_fn_state(xs, g_xs)

        # numeric grad
        eps = 1e-4
        expected_grad = np.zeros((1, 3))
        for i in range(3):
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] + eps
            forward = \
                TwoWheeledConfigModule.state_cost_fn(tmp_x, g_xs)
            forward = np.sum(forward)
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] - eps
            backward = \
                TwoWheeledConfigModule.state_cost_fn(tmp_x, g_xs)
            backward = np.sum(backward)

            expected_grad[0, i] = (forward - backward) / (2. * eps)

        assert cost_grad == pytest.approx(expected_grad)

    def test_input_gradient(self):
        """
        """
        us = np.ones((1, 2))
        cost_grad =\
            TwoWheeledConfigModule.gradient_cost_fn_input(None, us)

        # numeric grad
        eps = 1e-4
        expected_grad = np.zeros((1, 2))
        for i in range(2):
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] + eps
            forward = \
                TwoWheeledConfigModule.input_cost_fn(tmp_u)
            forward = np.sum(forward)
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] - eps
            backward = \
                TwoWheeledConfigModule.input_cost_fn(tmp_u)
            backward = np.sum(backward)

            expected_grad[0, i] = (forward - backward) / (2. * eps)

        assert cost_grad == pytest.approx(expected_grad)

    def test_state_hessian(self):
        """
        """
        g_xs = np.ones((1, 3)) * 0.5
        xs = np.ones((1, 3))
        cost_hess =\
            TwoWheeledConfigModule.hessian_cost_fn_state(xs, g_xs)

        # numeric grad
        eps = 1e-4
        expected_hess = np.zeros((1, 3, 3))
        for i in range(3):
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] + eps
            forward = \
                TwoWheeledConfigModule.gradient_cost_fn_state(
                    tmp_x, g_xs, terminal=False)
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] - eps
            backward = \
                TwoWheeledConfigModule.gradient_cost_fn_state(
                    tmp_x, g_xs, terminal=False)

            expected_hess[0, :, i] = (forward - backward) / (2. * eps)

        assert cost_hess == pytest.approx(expected_hess)

    def test_input_hessian(self):
        """
        """
        us = np.ones((1, 2))
        xs = np.ones((1, 3))
        cost_hess = TwoWheeledConfigModule.hessian_cost_fn_input(us, xs)

        # numeric grad
        eps = 1e-4
        expected_hess = np.zeros((1, 2, 2))
        for i in range(2):
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] + eps
            forward = \
                TwoWheeledConfigModule.gradient_cost_fn_input(
                    xs, tmp_u)
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] - eps
            backward = \
                TwoWheeledConfigModule.gradient_cost_fn_input(
                    xs, tmp_u)

            expected_hess[0, :, i] = (forward - backward) / (2. * eps)

        assert cost_hess == pytest.approx(expected_hess)
