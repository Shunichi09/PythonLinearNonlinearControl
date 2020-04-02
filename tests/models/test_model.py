import pytest
import numpy as np

from PythonLinearNonlinearControl.models.model import LinearModel

class TestLinearModel():
    """
    """
    def test_predict(self):
    
        A = np.array([[1., 0.1],
                      [0.1, 1.5]])
        B = np.array([[0.2], [0.5]])
        curr_x = np.ones(2) * 0.5
        u = np.ones((1, 1))

        linear_model = LinearModel(A, B)
        pred_xs = linear_model.predict_traj(curr_x, u)

        assert pred_xs == pytest.approx(np.array([[0.5, 0.5], [0.75, 1.3]]))
    
    def test_alltogether(self):

        A = np.array([[1., 0.1],
                      [0.1, 1.5]])
        B = np.array([[0.2], [0.5]])
        curr_x = np.ones(2) * 0.5
        u = np.ones((1, 1))

        linear_model = LinearModel(A, B)
        pred_xs = linear_model.predict_traj(curr_x, u)

        u = np.tile(u, (1, 1, 1))
        pred_xs_alltogether = linear_model.predict_traj(curr_x, u)[0]

        assert pred_xs_alltogether == pytest.approx(pred_xs)

    def test_alltogether_val(self):

        A = np.array([[1., 0.1],
                      [0.1, 1.5]])
        B = np.array([[0.2], [0.5]])
        curr_x = np.ones(2) * 0.5
        u = np.stack((np.ones((1, 1)), np.ones((1, 1))*0.5), axis=0)

        linear_model = LinearModel(A, B)

        pred_xs_alltogether = linear_model.predict_traj(curr_x, u)
        
        expected_val = np.array([[[0.5, 0.5], [0.75, 1.3]],
                                 [[0.5, 0.5], [0.65, 1.05]]])

        assert pred_xs_alltogether == pytest.approx(expected_val)