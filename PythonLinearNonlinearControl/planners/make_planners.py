from .const_planner import ConstantPlanner
from .closest_point_planner import ClosestPointPlanner


def make_planner(args, config):

    if args.env == "FirstOrderLag":
        return ConstantPlanner(config)
    elif args.env == "TwoWheeledConst":
        return ConstantPlanner(config)
    elif args.env == "TwoWheeledTrack":
        return ClosestPointPlanner(config)
    elif args.env == "CartPole":
        return ConstantPlanner(config)
    elif args.env == "NonlinearSample":
        return ConstantPlanner(config)

    raise NotImplementedError(
        "There is not {} Planner".format(args.planner_type))
