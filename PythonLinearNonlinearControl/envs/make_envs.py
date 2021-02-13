from .first_order_lag import FirstOrderLagEnv
from .two_wheeled import TwoWheeledConstEnv
from .two_wheeled import TwoWheeledTrackEnv
from .cartpole import CartPoleEnv
from .nonlinear_sample_system import NonlinearSampleSystemEnv


def make_env(args):

    if args.env == "FirstOrderLag":
        return FirstOrderLagEnv()
    elif args.env == "TwoWheeledConst":
        return TwoWheeledConstEnv()
    elif args.env == "TwoWheeledTrack":
        return TwoWheeledTrackEnv()
    elif args.env == "CartPole":
        return CartPoleEnv()
    elif args.env == "NonlinearSample":
        return NonlinearSampleSystemEnv()

    raise NotImplementedError("There is not {} Env".format(args.env))
