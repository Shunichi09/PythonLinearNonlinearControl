from .first_order_lag import FirstOrderLagConfigModule

def make_config(args):
    """
    Returns:
        config (ConfigModule class): configuration for the each env
    """
    if args.env == "FirstOrderLag":
        return FirstOrderLagConfigModule()