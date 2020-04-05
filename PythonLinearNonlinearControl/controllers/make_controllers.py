from .mpc import LinearMPC
from .cem import CEM
from .random import RandomShooting
from .mppi import MPPI
from .ilqr import iLQR
from .ddp import DDP

def make_controller(args, config, model):

    if args.controller_type == "MPC":
        return LinearMPC(config, model)
    elif args.controller_type == "CEM":
        return CEM(config, model)
    elif args.controller_type == "Random":
        return RandomShooting(config, model)
    elif args.controller_type == "MPPI":
        return MPPI(config, model)
    elif args.controller_type == "iLQR":
        return iLQR(config, model)
    elif args.controller_type == "DDP":
        return DDP(config, model)