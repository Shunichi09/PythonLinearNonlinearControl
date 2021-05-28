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
    make_logger(args.result_dir)

    env = make_env(args)

    config = make_config(args)

    planner = make_planner(args, config)

    model = make_model(args, config)

    controller = make_controller(args, config, model)

    runner = make_runner(args)

    history_x, history_u, history_g = runner.run(env, controller, planner)

    plot_results(history_x, history_u, history_g=history_g, args=args)
    save_plot_data(history_x, history_u, history_g=history_g, args=args)

    if args.save_anim:
        animator = Animator(env, args=args)
        animator.draw(history_x, history_g)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller_type", type=str, default="NMPCCGMRES")
    parser.add_argument("--env", type=str, default="TwoWheeledConst")
    parser.add_argument("--save_anim", type=bool_flag, default=0)
    parser.add_argument("--result_dir", type=str, default="./result")

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
