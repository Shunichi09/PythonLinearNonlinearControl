import pytest
import numpy as np

from PythonLinearNonlinearControl.models.cartpole import CartPoleModel
from PythonLinearNonlinearControl.configs.cartpole \
    import CartPoleConfigModule
    
class TestCartPoleModel():
    """
    """
    def test_step(self):
        config = CartPoleConfigModule()
        cartpole_model = CartPoleModel(config)

        curr_x = np.ones(4)
        curr_x[2] = np.pi / 6.

        us = np.ones((1, 1))

        next_x = cartpole_model.predict_traj(curr_x, us)

        d_x0 = curr_x[1]
        d_x1 = (1. + config.MP * np.sin(np.pi / 6.) \
                     * (config.L * (1.**2) \
                        + config.G * np.cos(np.pi / 6.))) \
                / (config.MC + config.MP * np.sin(np.pi / 6.)**2)
        d_x2 = curr_x[3]
        d_x3 = (-1. * np.cos(np.pi / 6.) \
                - config.MP * config.L * (1.**2) \
                  * np.cos(np.pi / 6.) * np.sin(np.pi / 6.) \
                - (config.MP + config.MC) * config.G \
                   * np.sin(np.pi / 6.)) \
                 / (config.L \
                     * (config.MC \
                        + config.MP * np.sin(np.pi / 6.)**2))

        expected = np.array([d_x0, d_x1, d_x2, d_x3]) * config.DT \
                   + curr_x

        expected = np.stack((curr_x, expected), axis=0)

        assert next_x == pytest.approx(expected, abs=1e-5) 

    def test_predict_traj(self):
        config = CartPoleConfigModule()
        cartpole_model =  CartPoleModel(config)

        curr_x = np.ones(config.STATE_SIZE)
        curr_x[-1] = np.pi / 6.
        u = np.ones((1, config.INPUT_SIZE))

        pred_xs = cartpole_model.predict_traj(curr_x, u)

        u = np.tile(u, (2, 1, 1))
        pred_xs_alltogether = cartpole_model.predict_traj(curr_x, u)[0]

        assert pred_xs_alltogether == pytest.approx(pred_xs)
    
    def test_gradient_state(self):
        
        config = CartPoleConfigModule()
        cartpole_model =  CartPoleModel(config)

        xs = np.ones((1, config.STATE_SIZE)) \
             * np.random.rand(1, config.STATE_SIZE)
        xs[0, -1] = np.pi / 6.
        us = np.ones((1, config.INPUT_SIZE))

        grad = cartpole_model.calc_f_x(xs, us, config.DT)

        # expected cost
        expected_grad = np.zeros((1, config.STATE_SIZE, config.STATE_SIZE))
        eps = 1e-4

        for i in range(config.STATE_SIZE):
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] + eps
            forward = \
                cartpole_model.predict_next_state(tmp_x[0], us[0])
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] - eps
            backward = \
                cartpole_model.predict_next_state(tmp_x[0], us[0])

            expected_grad[0, :, i] = (forward - backward) / (2. * eps)
        
        assert grad == pytest.approx(expected_grad)
    
    def test_gradient_input(self):
        
        config = CartPoleConfigModule()
        cartpole_model =  CartPoleModel(config)

        xs = np.ones((1, config.STATE_SIZE)) \
             * np.random.rand(1, config.STATE_SIZE)
        xs[0, -1] = np.pi / 6.
        us = np.ones((1, config.INPUT_SIZE))

        grad = cartpole_model.calc_f_u(xs, us, config.DT)

        # expected cost
        expected_grad = np.zeros((1, config.STATE_SIZE, config.INPUT_SIZE))
        eps = 1e-4

        for i in range(config.INPUT_SIZE):
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] + eps
            forward = \
                cartpole_model.predict_next_state(xs[0], tmp_u[0])
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] - eps
            backward = \
                cartpole_model.predict_next_state(xs[0], tmp_u[0])

            expected_grad[0, :, i] = (forward - backward) / (2. * eps)
        
        assert grad == pytest.approx(expected_grad)
