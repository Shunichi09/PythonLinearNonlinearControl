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

        costs = config.terminal_state_cost_fn(pred_xs[:, -1, :],
                                              g_xs[:, -1, :])

        assert costs.shape == (pop_size, 1)


class TestGradient():
    def test_state_gradient(self):
        """
        """
        xs = np.ones((1, 4))
        cost_grad = CartPoleConfigModule.gradient_cost_fn_state(xs, None)

        # numeric grad
        eps = 1e-4
        expected_grad = np.zeros((1, 4))
        for i in range(4):
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] + eps
            forward = \
                CartPoleConfigModule.state_cost_fn(tmp_x, None)
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] - eps
            backward = \
                CartPoleConfigModule.state_cost_fn(tmp_x, None)

            expected_grad[0, i] = (forward - backward) / (2. * eps)

        assert cost_grad == pytest.approx(expected_grad)

    def test_terminal_state_gradient(self):
        """
        """
        xs = np.ones(4)
        cost_grad =\
            CartPoleConfigModule.gradient_cost_fn_state(xs, None,
                                                        terminal=True)

        # numeric grad
        eps = 1e-4
        expected_grad = np.zeros((1, 4))
        for i in range(4):
            tmp_x = xs.copy()
            tmp_x[i] = xs[i] + eps
            forward = \
                CartPoleConfigModule.state_cost_fn(tmp_x, None)
            tmp_x = xs.copy()
            tmp_x[i] = xs[i] - eps
            backward = \
                CartPoleConfigModule.state_cost_fn(tmp_x, None)

            expected_grad[0, i] = (forward - backward) / (2. * eps)

        assert cost_grad == pytest.approx(expected_grad)

    def test_input_gradient(self):
        """
        """
        us = np.ones((1, 1))
        cost_grad = CartPoleConfigModule.gradient_cost_fn_input(None, us)

        # numeric grad
        eps = 1e-4
        expected_grad = np.zeros((1, 1))
        for i in range(1):
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] + eps
            forward = \
                CartPoleConfigModule.input_cost_fn(tmp_u)
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] - eps
            backward = \
                CartPoleConfigModule.input_cost_fn(tmp_u)

            expected_grad[0, i] = (forward - backward) / (2. * eps)

        assert cost_grad == pytest.approx(expected_grad)

    def test_state_hessian(self):
        """
        """
        xs = np.ones((1, 4))
        cost_hess = CartPoleConfigModule.hessian_cost_fn_state(xs, None)

        # numeric grad
        eps = 1e-4
        expected_hess = np.zeros((1, 4, 4))
        for i in range(4):
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] + eps
            forward = \
                CartPoleConfigModule.gradient_cost_fn_state(
                    tmp_x, None, terminal=False)
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] - eps
            backward = \
                CartPoleConfigModule.gradient_cost_fn_state(
                    tmp_x, None, terminal=False)

            expected_hess[0, :, i] = (forward - backward) / (2. * eps)

        assert cost_hess == pytest.approx(expected_hess)

    def test_terminal_state_hessian(self):
        """
        """
        xs = np.ones(4)
        cost_hess =\
            CartPoleConfigModule.hessian_cost_fn_state(xs, None,
                                                       terminal=True)

        # numeric grad
        eps = 1e-4
        expected_hess = np.zeros((1, 4, 4))
        for i in range(4):
            tmp_x = xs.copy()
            tmp_x[i] = xs[i] + eps
            forward = \
                CartPoleConfigModule.gradient_cost_fn_state(
                    tmp_x, None, terminal=True)
            tmp_x = xs.copy()
            tmp_x[i] = xs[i] - eps
            backward = \
                CartPoleConfigModule.gradient_cost_fn_state(
                    tmp_x, None, terminal=True)

            expected_hess[0, :, i] = (forward - backward) / (2. * eps)

        assert cost_hess == pytest.approx(expected_hess)

    def test_input_hessian(self):
        """
        """
        us = np.ones((1, 1))
        xs = np.ones((1, 4))
        cost_hess = CartPoleConfigModule.hessian_cost_fn_input(us, xs)

        # numeric grad
        eps = 1e-4
        expected_hess = np.zeros((1, 1, 1))
        for i in range(1):
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] + eps
            forward = \
                CartPoleConfigModule.gradient_cost_fn_input(
                    xs, tmp_u)
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] - eps
            backward = \
                CartPoleConfigModule.gradient_cost_fn_input(
                    xs, tmp_u)

            expected_hess[0, :, i] = (forward - backward) / (2. * eps)

        assert cost_hess == pytest.approx(expected_hess)
