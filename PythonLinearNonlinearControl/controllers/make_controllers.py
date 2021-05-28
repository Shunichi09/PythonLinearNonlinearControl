from .mpc import LinearMPC
from .cem import CEM
from .random import RandomShooting
from .mppi import MPPI
from .mppi_williams import MPPIWilliams
from .ilqr import iLQR
from .ddp import DDP
from .nmpc import NMPC
from .nmpc_cgmres import NMPCCGMRES


def make_controller(args, config, model):

    if args.controller_type == "MPC":
        return LinearMPC(config, model)
    elif args.controller_type == "CEM":
        return CEM(config, model)
    elif args.controller_type == "Random":
        return RandomShooting(config, model)
    elif args.controller_type == "MPPI":
        return MPPI(config, model)
    elif args.controller_type == "MPPIWilliams":
        return MPPIWilliams(config, model)
    elif args.controller_type == "iLQR":
        return iLQR(config, model)
    elif args.controller_type == "DDP":
        return DDP(config, model)
    elif args.controller_type == "NMPC":
        return NMPC(config, model)
    elif args.controller_type == "NMPCCGMRES":
        return NMPCCGMRES(config, model)

    raise ValueError("No controller: {}".format(args.controller_type))
