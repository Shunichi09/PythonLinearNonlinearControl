from .first_order_lag import FirstOrderLagConfigModule
from .two_wheeled import TwoWheeledConfigModule, TwoWheeledExtendConfigModule
from .cartpole import CartPoleConfigModule
from .nonlinear_sample_system import NonlinearSampleSystemConfigModule, NonlinearSampleSystemExtendConfigModule


def make_config(args):
    """
    Returns:
        config (ConfigModule class): configuration for the each env
    """
    if args.env == "FirstOrderLag":
        return FirstOrderLagConfigModule()
    elif args.env == "TwoWheeledConst" or args.env == "TwoWheeledTrack":
        if args.controller_type == "NMPCCGMRES":
            return TwoWheeledExtendConfigModule()
        return TwoWheeledConfigModule()
    elif args.env == "CartPole":
        return CartPoleConfigModule()
    elif args.env == "NonlinearSample":
        if args.controller_type == "NMPCCGMRES":
            return NonlinearSampleSystemExtendConfigModule()
        return NonlinearSampleSystemConfigModule()
