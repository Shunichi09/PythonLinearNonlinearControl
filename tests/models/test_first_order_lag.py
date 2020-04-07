import pytest
import numpy as np

from PythonLinearNonlinearControl.models.model \
    import LinearModel
from PythonLinearNonlinearControl.models.first_order_lag \
    import FirstOrderLagModel
from PythonLinearNonlinearControl.configs.first_order_lag \
    import FirstOrderLagConfigModule

from unittest.mock import patch
from unittest.mock import Mock

class TestFirstOrderLagModel():
    """
    """
    def test_step(self):
        config = FirstOrderLagConfigModule()
        firstorderlag_model = FirstOrderLagModel(config)

        curr_x = np.ones(config.STATE_SIZE)
        u = np.ones((1, config.INPUT_SIZE))

        with patch.object(LinearModel, "predict_traj") as mock_predict_traj:
            firstorderlag_model.predict_traj(curr_x, u)
            
            mock_predict_traj.assert_called_once_with(curr_x, u)
    
    def test_predict_traj(self):

        config = FirstOrderLagConfigModule()
        firstorderlag_model = FirstOrderLagModel(config)

        curr_x = np.ones(config.STATE_SIZE)
        curr_x[-1] = np.pi / 6.
        u = np.ones((1, config.INPUT_SIZE))

        pred_xs = firstorderlag_model.predict_traj(curr_x, u)

        u = np.tile(u, (1, 1, 1))
        pred_xs_alltogether = firstorderlag_model.predict_traj(curr_x, u)[0]

        assert pred_xs_alltogether == pytest.approx(pred_xs)