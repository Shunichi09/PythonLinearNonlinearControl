import argparse

from PythonLinearNonlinearControl.helper import bool_flag, make_logger
from PythonLinearNonlinearControl.controllers.make_controllers import make_controller
from PythonLinearNonlinearControl.planners.make_planners import make_planner
from PythonLinearNonlinearControl.configs.make_configs import make_config
from PythonLinearNonlinearControl.models.make_models import make_model
from PythonLinearNonlinearControl.envs.make_envs import make_env
from PythonLinearNonlinearControl.runners.make_runners import make_runner
from PythonLinearNonlinearControl.plotters.plot_func import plot_results, \
                                                            save_plot_data
from PythonLinearNonlinearControl.plotters.animator import Animator

def run(args):
    # logger
    make_logger(args.result_dir)

    # make envs
    env = make_env(args)

    # make config 
    config = make_config(args)

    # make planner
    planner = make_planner(args, config)
    
    # make model
    model = make_model(args, config)
    
    # make controller
    controller = make_controller(args, config, model)
    
    # make simulator
    runner = make_runner(args)

    # run experiment
    history_x, history_u, history_g = runner.run(env, controller, planner) 

    # plot results
    plot_results(args, history_x, history_u, history_g=history_g)
    save_plot_data(args, history_x, history_u, history_g=history_g)

    if args.save_anim:
        animator = Animator(env, args=args)
        animator.draw(history_x, history_g)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller_type", type=str, default="CEM")
    parser.add_argument("--env", type=str, default="TwoWheeledTrack")
    parser.add_argument("--save_anim", type=bool_flag, default=1)
    parser.add_argument("--result_dir", type=str, default="./result")

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()