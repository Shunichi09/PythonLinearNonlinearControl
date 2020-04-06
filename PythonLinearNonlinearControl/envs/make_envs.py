from .first_order_lag import FirstOrderLagEnv
from .two_wheeled import TwoWheeledConstEnv
from .cartpole import CartpoleEnv

def make_env(args):

    if args.env == "FirstOrderLag":
        return FirstOrderLagEnv()
    elif args.env == "TwoWheeledConst":
        return TwoWheeledConstEnv()
    elif args.env == "CartPole":
        return CartpoleEnv()
    
    raise NotImplementedError("There is not {} Env".format(args.env))