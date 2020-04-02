from .first_order_lag import FirstOrderLagEnv

def make_env(args):

    if args.env == "FirstOrderLag":
        return FirstOrderLagEnv()
    
    raise NotImplementedError("There is not {} Env".format(name))