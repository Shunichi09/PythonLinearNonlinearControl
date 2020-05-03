# PythonLinearnNonlinearControl Quickstart Guide

This is a quickstart guide for users who just want to try PythonLinearNonlinearControl.
If you have not installed PythonLinearNonLinearControl, please see the section of "how to setup" in README.md

When we design control systems, we should have Environment, Model, Planner, Controller and Runner.
Therefore your script contains those Modules.

First, import each Modules from PythonLinearNonlinearControl.

```py
from PythonLinearNonlinearControl import configs 
from PythonLinearNonlinearControl import envs
from PythonLinearNonlinearControl import models
from PythonLinearNonlinearControl import planners
from PythonLinearNonlinearControl import controllers
from PythonLinearNonlinearControl import runners
```

Configs contains each modules configurations such as cost functions, prediction length, ...etc.

Then you can make each module. (This is example about CEM and CartPole env)

```py
config = configs.CartPoleConfigModule()
env = envs.CartPoleEnv()
model = models.CartPoleModel(config)
planner = controllers.CEM(config, model)
runner = planners.ConstantPlanner(config)
controller = runners.ExpRunner()
```

The preparation for experiment has done!
Please run the runner.

```py
history_x, history_u, history_g = runner.run(env, controller, planner) 
```

You can get the results of history of state, history of input and history of goal.
Use that histories to visualize the Animation or Figures.
(Note FirstOrderEnv does not support animation)

```py
# plot results
plot_results(args, history_x, history_u, history_g=history_g)
save_plot_data(args, history_x, history_u, history_g=history_g)

# create animation
animator = Animator(args, env)
animator.draw(history_x, history_g)
```