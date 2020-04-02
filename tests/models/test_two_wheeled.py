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