from .const_planner import ConstantPlanner

def make_planner(args, config):
    
    if args.planner_type == "const":
        return ConstantPlanner(config)
    
    raise NotImplementedError("There is not {} Planner".format(args.planner_type))