[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/Shunichi09/PythonLinearNonlinearControl/badge.svg?branch=master)](https://coveralls.io/github/Shunichi09/PythonLinearNonlinearControl?branch=master)
[![Build Status](https://travis-ci.org/Shunichi09/PythonLinearNonlinearControl.svg?branch=master)](https://travis-ci.org/Shunichi09/PythonLinearNonlinearControl)

# PythonLinearNonLinearControl

PythonLinearNonLinearControl is a library implementing the linear and nonlinear control theories in python.

![Concepts](assets/concepts.png)

# Algorithms

| Algorithm | Use Linear Model | Use Nonlinear Model | Need Gradient (Hamiltonian) | Need Gradient (Model) |
|:----------|:---------------: |:----------------:|:----------------:|:----------------:|
| Linear Model Predictive Control (MPC) | ✓ | x | x | x | 
| Cross Entropy Method (CEM) | ✓ | ✓ | x | x | 
| Model Preidictive Path Integral Control (MPPI) | ✓ | ✓ | x | x | 
| Random Shooting Method (Random) | ✓ | ✓ | x | x | 
| Iterative LQR (iLQR) | x | ✓ | x | ✓ | 
| Unconstrained Nonlinear Model Predictive Control (NMPC) | x | ✓ | ✓ | x |
| Constrained Nonlinear Model Predictive Control CGMRES (NMPC-CGMRES) | x | ✓ | ✓ | x |
| Constrained Nonlinear Model Predictive Control Newton (NMPC-Newton) | x | ✓ | x | x |

"Need Gradient" means that you have to implement the gradient of the model or the gradient of hamiltonian.  
This library is also easily to extend for your own situations.

Following algorithms are implemented in PythonLinearNonlinearControl

- [Linear Model Predictive Control (MPC)](http://www2.eng.cam.ac.uk/~jmm1/mpcbook/mpcbook.html)
  - Ref: Maciejowski, J. M. (2002). Predictive control: with constraints.
    - [script]()
- [Cross Entropy Method (CEM)](https://arxiv.org/abs/1805.12114)
  - Ref: Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Advances in Neural Information Processing Systems (pp. 4754-4765)
    - [script]()
- [Model Preidictive Path Integral Control (MPPI)](https://arxiv.org/abs/1909.11652)
  - Ref: Nagabandi, A., Konoglie, K., Levine, S., & Kumar, V. (2019). Deep Dynamics Models for Learning Dexterous Manipulation. arXiv preprint arXiv:1909.11652.
    - [script]()
- [Random Shooting Method (Random)](https://arxiv.org/abs/1805.12114)
  - Ref: Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Advances in Neural Information Processing Systems (pp. 4754-4765)
    - [script]()
- [Iterative LQR (iLQR)](https://ieeexplore.ieee.org/document/6386025)
  - Ref: Tassa, Y., Erez, T., & Todorov, E. (2012, October). Synthesis and stabilization of complex behaviors through online trajectory optimization. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 4906-4913). IEEE. and [Study Wolf](https://github.com/studywolf/control)
    - [script (Coming soon)]()
- [Unconstrained Nonlinear Model Predictive Control (NMPC)](https://www.sciencedirect.com/science/article/pii/S0005109897000058)
  - Ref: Ohtsuka, T., & Fujii, H. A. (1997). Real-time optimization algorithm for nonlinear receding-horizon control. Automatica, 33(6), 1147-1154.
    - [script (Coming soon)]()
- [Constrained Nonlinear Model Predictive Control -CGMRES- (NMPC-CGMRES)](https://www.sciencedirect.com/science/article/pii/S0005109897000058)
  - Ref: Ohtsuka, T., & Fujii, H. A. (1997). Real-time optimization algorithm for nonlinear receding-horizon control. Automatica, 33(6), 1147-1154.
    - [script (Coming soon)]()
- [Constrained Nonlinear Model Predictive Control -Newton- (NMPC-Newton)](https://www.sciencedirect.com/science/article/pii/S0005109897000058)
  - Ref: Ohtsuka, T., & Fujii, H. A. (1997). Real-time optimization algorithm for nonlinear receding-horizon control. Automatica, 33(6), 1147-1154.
    - [script (Coming soon)]()

# Environments

| Name | Linear | Nonlinear | State Size | Input size |
|:----------|:---------------:|:----------------:|:----------------:|:----------------:|
| First Order Lag System | ✓ | x | 4 | 2 | 
| Two wheeled System (Constant Goal) | x | ✓ | 3 | 2 | 
| Two wheeled System (Moving Goal) (Coming soon) | x | ✓ | 3 | 2 | 

All environments are continuous.
**It should be noted that the algorithms for linear model could be applied to nonlinear enviroments if you have linealized the model of nonlinear environments.**

# Usage 

## To install this package

```
python setup.py install
```

or 

```
pip install .
```

## When developing the package

```
python setup.py develop
```

or

```
pip install -e .
```

## Run Experiments

You can run the experiments as follows:

```
python scripts/simple_run.py --model "first-order_lag" --controller "CEM"
```

**figures and animations are saved in the ./result folder.**

# Basic concepts

When we design control systems, we should have **Model**, **Planner**, **Controller** and **Runner** as shown in the figure.
It should be noted that **Model** and **Environment** are different. As mentioned before, we the algorithms for linear model could be applied to nonlinear enviroments if you have linealized model of nonlinear environments. In addition, you can use Neural Network or any non-linear functions to the model, although this library can not deal with it now.

![Concepts](assets/concepts.png)

## Model

System model. For an instance, in the case that a model is linear, this model should have a form, "x[k+1] = Ax[k] + Bu[k]".

If you use gradient based control method, you are preferred to implement the gradients of the model, other wise the controllers use numeric gradients.

## Planner

Planner make the goal states.

## Controller

Controller calculate the optimal inputs by using the model by using the algorithms.

## Runner

Runner runs the simulation.

Please, see more detail in each scripts.

# Old version

If you are interested in the old version of this library, that was not a library just examples, please see [v1.0](https://github.com/Shunichi09/PythonLinearNonlinearControl/tree/v1.0)

# Documents

Coming soon !!

# Requirements

- numpy
- matplotlib
- cvxopt
- scipy

# License

[MIT License](LICENSE).

# Citation

```
@Misc{PythonLinearNonLinearControl,
author = {Shunichi Sekiguchi},
title = {PythonLinearNonlinearControl},
note = "\url{https://github.com/Shunichi09/PythonLinearNonlinearControl}",
}
```
