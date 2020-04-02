from .first_order_lag import FirstOrderLagEnv
from .two_wheeled import TwoWheeledConstEnv

def make_env(args):

    if args.env == "FirstOrderLag":
        return FirstOrderLagEnv()
    elif args.env == "TwoWheeledConst":
        return TwoWheeledConstEnv()
    
    raise NotImplementedError("There is not {} Env".format(args.env))