from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)

class iLQR(Controller):
    """ iterative Liner Quadratique Regulator
    """
    def __init__(self, config, model):
        """
        """
        super(iLQR, self).__init__(config, model)
            
        if config.TYPE != "Nonlinear":
            raise ValueError("{} could be not applied to \
                              this controller".format(model))

        self.model = model
        