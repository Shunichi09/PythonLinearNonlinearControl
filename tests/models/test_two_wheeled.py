import pytest
import numpy as np

from PythonLinearNonlinearControl.models.two_wheeled import TwoWheeledModel
from PythonLinearNonlinearControl.configs.two_wheeled \
    import TwoWheeledConfigModule
    
class TestTwoWheeledModel():
    """
    """
    def test_step(self):
        config = TwoWheeledConfigModule()
        two_wheeled_model = TwoWheeledModel(config)

        curr_x = np.ones(config.STATE_SIZE)
        curr_x[-1] = np.pi / 6.
        u = np.ones((1, config.INPUT_SIZE))

        next_x = two_wheeled_model.predict_traj(curr_x, u)

        pos_x = np.cos(curr_x[-1]) * u[0, 0] * config.DT + curr_x[0]
        pos_y = np.sin(curr_x[-1]) * u[0, 0] * config.DT + curr_x[1]

        expected = np.array([[1., 1., np.pi / 6.],
                             [pos_x, pos_y, curr_x[-1] + u[0, 1] * config.DT]])

        assert next_x == pytest.approx(expected) 
    
    def test_predict_traj(self):
        config = TwoWheeledConfigModule()
        two_wheeled_model = TwoWheeledModel(config)

        curr_x = np.ones(config.STATE_SIZE)
        curr_x[-1] = np.pi / 6.
        u = np.ones((1, config.INPUT_SIZE))

        pred_xs = two_wheeled_model.predict_traj(curr_x, u)

        u = np.tile(u, (1, 1, 1))
        pred_xs_alltogether = two_wheeled_model.predict_traj(curr_x, u)[0]

        assert pred_xs_alltogether == pytest.approx(pred_xs)
    
    def test_gradient_state(self):
        
        config = TwoWheeledConfigModule()
        two_wheeled_model = TwoWheeledModel(config)

        xs = np.ones((1, config.STATE_SIZE))
        xs[0, -1] = np.pi / 6.
        us = np.ones((1, config.INPUT_SIZE))

        grad = two_wheeled_model.calc_f_x(xs, us, config.DT)

        # expected cost
        expected_grad = np.zeros((1, config.STATE_SIZE, config.STATE_SIZE))
        eps = 1e-4

        for i in range(config.STATE_SIZE):
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] + eps
            forward = \
                two_wheeled_model.predict_next_state(tmp_x[0], us[0])
            tmp_x = xs.copy()
            tmp_x[0, i] = xs[0, i] - eps
            backward = \
                two_wheeled_model.predict_next_state(tmp_x[0], us[0])

            expected_grad[0, :, i] = (forward - backward) / (2. * eps)
        
        assert grad == pytest.approx(expected_grad)
    
    def test_gradient_input(self):
        
        config = TwoWheeledConfigModule()
        two_wheeled_model = TwoWheeledModel(config)

        xs = np.ones((1, config.STATE_SIZE))
        xs[0, -1] = np.pi / 6.
        us = np.ones((1, config.INPUT_SIZE))

        grad = two_wheeled_model.calc_f_u(xs, us, config.DT)

        # expected cost
        expected_grad = np.zeros((1, config.STATE_SIZE, config.INPUT_SIZE))
        eps = 1e-4

        for i in range(config.INPUT_SIZE):
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] + eps
            forward = \
                two_wheeled_model.predict_next_state(xs[0], tmp_u[0])
            tmp_u = us.copy()
            tmp_u[0, i] = us[0, i] - eps
            backward = \
                two_wheeled_model.predict_next_state(xs[0], tmp_u[0])

            expected_grad[0, :, i] = (forward - backward) / (2. * eps)
        
        assert grad == pytest.approx(expected_grad)
