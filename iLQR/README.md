# Iterative Linear Quadratic Regulator
This program is about iLQR (Iteratice Linear Quadratic Regulator)

# Problem Formulation

Two wheeled robot is expressed by the following equation.
It is nonlinear and nonholonomic system. Sometimes, it's extremely difficult to control the 
steering(or angular velocity) and velocity of the wheeled robot. Therefore, many methods control only steering, like purepersuit, Linear MPC.
However, sometimes we should consider the velocity and steering simultaneously when the car or robots move fast.
To solve the problem, we should apply the control methods which can treat the nonlinear system.

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{dt}&space;\boldsymbol{X}=&space;\frac{d}{dt}&space;\begin{bmatrix}&space;x&space;\\&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;\cos(\theta)&space;&&space;0&space;\\&space;\sin(\theta)&space;&&space;0&space;\\&space;0&space;&&space;1&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_v&space;\\&space;u_\omega&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{B}\boldsymbol{U}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dt}&space;\boldsymbol{X}=&space;\frac{d}{dt}&space;\begin{bmatrix}&space;x&space;\\&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;\cos(\theta)&space;&&space;0&space;\\&space;\sin(\theta)&space;&&space;0&space;\\&space;0&space;&&space;1&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_v&space;\\&space;u_\omega&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{B}\boldsymbol{U}" title="\frac{d}{dt} \boldsymbol{X}= \frac{d}{dt} \begin{bmatrix} x \\ y \\ \theta \end{bmatrix} = \begin{bmatrix} \cos(\theta) & 0 \\ \sin(\theta) & 0 \\ 0 & 1 \\ \end{bmatrix} \begin{bmatrix} u_v \\ u_\omega \\ \end{bmatrix} = \boldsymbol{B}\boldsymbol{U}" /></a>

Nonliner Model Predictive Control is one of the famous methods, so I applied the method to two-wheeled robot which is included in the folder of this repository.
(if you are interested, please go to nmpc/ folder of this repository)

NMPC is very effecitive method to solve nonlinear optimal control problem but it is a handcraft method.
This program is about one more other methods to solve the nonlinear optimal control problem.

The method is iterative LQR.
Iterative LQR is one of the DDP(differential dynamic programming) methods.
Recently, this method is used in model-based RL(reinforcement learning).
Although, this method cannot guarantee to obtain the global optimal answer, we could apply any model such as nonliner model or time-varing model even the model that expressed by NN.
(Still we can only get approximate optimal anwser)

If you want to know more about the iLQR, please look the references.
The paper and website are great.

# Usage

## static goal

```
$ python3 main_static.py
```

## dynamic goal

```
$ python3 main_dynamic.py
```

# Expected Results

- static goal




- track the goal





# Requirement

- python3.5 or more
- numpy
- matplotlib

# Reference

- study wolf
https://github.com/studywolf/control

- Sergey Levine's lecture
http://rail.eecs.berkeley.edu/deeprlcourse/

- Tassa, Y., Erez, T., & Todorov, E. (2012). Synthesis and stabilization of complex behaviors through online trajectory optimization. IEEE International Conference on Intelligent Robots and Systems, 4906–4913. https://doi.org/10.1109/IROS.2012.6386025

- Li, W., & Todorov, E. (n.d.). Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems. Retrieved from https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf