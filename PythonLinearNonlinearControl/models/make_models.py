from .first_order_lag import FirstOrderLagModel

def make_model(args, config):
    
    if args.env == "FirstOrderLag":
        return FirstOrderLagModel(config)
    
    raise NotImplementedError("There is not {} Model".format(args.env))
