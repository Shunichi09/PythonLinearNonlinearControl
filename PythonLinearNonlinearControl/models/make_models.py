from .first_order_lag import FirstOrderLagModel
from .two_wheeled import TwoWheeledModel

def make_model(args, config):
    
    if args.env == "FirstOrderLag":
        return FirstOrderLagModel(config)
    elif args.env == "TwoWheeledConst" or args.env == "TwoWheeled":
        return TwoWheeledModel(config)
    
    raise NotImplementedError("There is not {} Model".format(args.env))
