# Model Predictive Control Basic Tool
This program is about template, generic function of linear model predictive control

# Documentation of the MPC function
Linear model predicitive control should have state equation.
So if you want to use this function, you should model the plant as state equation.
Therefore, the parameters of this class are as following

**class MpcController()**

Attributes :

- A : numpy.ndarray
     - system matrix
- B : numpy.ndarray
    - input matrix
- Q : numpy.ndarray
    - evaluation function weight for states
- Qs : numpy.ndarray
    - concatenated evaluation function weight for states
- R : numpy.ndarray
    - evaluation function weight for inputs
- Rs : numpy.ndarray
    - concatenated evaluation function weight for inputs
- pre_step : int
    - prediction step
- state_size : int
    - state size of the plant
- input_size : int
    - input size of the plant
- dt_input_upper : numpy.ndarray, shape(input_size, ), optional
    - constraints of input dt, default is None
- dt_input_lower : numpy.ndarray, shape(input_size, ), optional
    - constraints of input dt, default is None
- input_upper : numpy.ndarray, shape(input_size, ), optional
    - constraints of input, default is None
- input_lower : numpy.ndarray, shape(input_size, ), optional
    - constraints of input, default is None

Methods:

- initialize_controller() initialize the controller
- calc_input(states, references) calculating optimal input

More details, please look the **mpc_func_with_scipy.py** and **mpc_func_with_cvxopt.py**

We have two function, mpc_func_with_cvxopt.py and mpc_func_with_scipy.py
Both functions have same variable and member function. However the solver is different. 
Plese choose the right method for your environment.

- example of import

```py
from mpc_func_with_scipy import MpcController as MpcController_scipy
from mpc_func_with_cvxopt import MpcController as MpcController_cvxopt
```

# Examples
## Problem Formulation

- **first order system**

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{dt}&space;\boldsymbol{X}&space;=&space;\begin{bmatrix}&space;-1/&space;\tau&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;-1/&space;\tau&space;&&space;0&space;&&space;0\\&space;1&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;1&space;&&space;0&space;&&space;0\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;v_x&space;\\&space;v_y&space;\\&space;x&space;\\&space;y&space;\end{bmatrix}&space;&plus;&space;\begin{bmatrix}&space;1/&space;\tau&space;&&space;0&space;\\&space;0&space;&&space;1/&space;\tau&space;\\&space;0&space;&&space;0&space;\\&space;0&space;&&space;0&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_x&space;\\&space;u_y&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{A}\boldsymbol{X}&space;&plus;&space;\boldsymbol{B}\boldsymbol{U}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dt}&space;\boldsymbol{X}&space;=&space;\begin{bmatrix}&space;-1/&space;\tau&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;-1/&space;\tau&space;&&space;0&space;&&space;0\\&space;1&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;1&space;&&space;0&space;&&space;0\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;v_x&space;\\&space;v_y&space;\\&space;x&space;\\&space;y&space;\end{bmatrix}&space;&plus;&space;\begin{bmatrix}&space;1/&space;\tau&space;&&space;0&space;\\&space;0&space;&&space;1/&space;\tau&space;\\&space;0&space;&&space;0&space;\\&space;0&space;&&space;0&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_x&space;\\&space;u_y&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{A}\boldsymbol{X}&space;&plus;&space;\boldsymbol{B}\boldsymbol{U}" title="\frac{d}{dt} \boldsymbol{X} = \begin{bmatrix} -1/ \tau & 0 & 0 & 0\\ 0 & -1/ \tau & 0 & 0\\ 1 & 0 & 0 & 0\\ 0 & 1 & 0 & 0\\ \end{bmatrix} \begin{bmatrix} v_x \\ v_y \\ x \\ y \end{bmatrix} + \begin{bmatrix} 1/ \tau & 0 \\ 0 & 1/ \tau \\ 0 & 0 \\ 0 & 0 \\ \end{bmatrix} \begin{bmatrix} u_x \\ u_y \\ \end{bmatrix} = \boldsymbol{A}\boldsymbol{X} + \boldsymbol{B}\boldsymbol{U}" /></a>

- **ACC (Adaptive cruise control)**

The two wheeled model are expressed the following equation.

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{dt}&space;\boldsymbol{X}=&space;\frac{d}{dt}&space;\begin{bmatrix}&space;x&space;\\&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;\cos(\theta)&space;&&space;0&space;\\&space;\sin(\theta)&space;&&space;0&space;\\&space;0&space;&&space;1&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_v&space;\\&space;u_\omega&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{B}\boldsymbol{U}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dt}&space;\boldsymbol{X}=&space;\frac{d}{dt}&space;\begin{bmatrix}&space;x&space;\\&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;\cos(\theta)&space;&&space;0&space;\\&space;\sin(\theta)&space;&&space;0&space;\\&space;0&space;&&space;1&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_v&space;\\&space;u_\omega&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{B}\boldsymbol{U}" title="\frac{d}{dt} \boldsymbol{X}= \frac{d}{dt} \begin{bmatrix} x \\ y \\ \theta \end{bmatrix} = \begin{bmatrix} \cos(\theta) & 0 \\ \sin(\theta) & 0 \\ 0 & 1 \\ \end{bmatrix} \begin{bmatrix} u_v \\ u_\omega \\ \end{bmatrix} = \boldsymbol{B}\boldsymbol{U}" /></a>

However, if we assume the velocity are constant, we can approximate the equation as following, 

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{dt}&space;\boldsymbol{X}=&space;\frac{d}{dt}&space;\begin{bmatrix}&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;0&space;&&space;V&space;\\&space;0&space;&&space;0&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;&plus;&space;\begin{bmatrix}&space;0&space;\\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_\omega&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{A}\boldsymbol{X}&space;&plus;&space;\boldsymbol{B}\boldsymbol{U}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dt}&space;\boldsymbol{X}=&space;\frac{d}{dt}&space;\begin{bmatrix}&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;0&space;&&space;V&space;\\&space;0&space;&&space;0&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;y&space;\\&space;\theta&space;\end{bmatrix}&space;&plus;&space;\begin{bmatrix}&space;0&space;\\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;u_\omega&space;\\&space;\end{bmatrix}&space;=&space;\boldsymbol{A}\boldsymbol{X}&space;&plus;&space;\boldsymbol{B}\boldsymbol{U}" title="\frac{d}{dt} \boldsymbol{X}= \frac{d}{dt} \begin{bmatrix} y \\ \theta \end{bmatrix} = \begin{bmatrix} 0 & V \\ 0 & 0 \\ \end{bmatrix} \begin{bmatrix} y \\ \theta \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} \begin{bmatrix} u_\omega \\ \end{bmatrix} = \boldsymbol{A}\boldsymbol{X} + \boldsymbol{B}\boldsymbol{U}" /></a>

then we can apply this model to linear mpc, we should give the model reference V although.

- **evaluation function**

the both examples have same evaluation function form as following equation.

<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;\sum_{i&space;=&space;0}^{prestep}||\boldsymbol{\hat{X}}(k&plus;i|k)-\boldsymbol{r}(k&plus;i|k)&space;||^2_{{\boldsymbol{Q}}(i)}&space;&plus;&space;||\boldsymbol{\Delta&space;{U}}(k&plus;i|k)||^2_{{\boldsymbol{R}}(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J&space;=&space;\sum_{i&space;=&space;0}^{prestep}||\boldsymbol{\hat{X}}(k&plus;i|k)-\boldsymbol{r}(k&plus;i|k)&space;||^2_{{\boldsymbol{Q}}(i)}&space;&plus;&space;||\boldsymbol{\Delta&space;{U}}(k&plus;i|k)||^2_{{\boldsymbol{R}}(i)}" title="J = \sum_{i = 0}^{prestep}||\boldsymbol{\hat{X}}(k+i|k)-\boldsymbol{r}(k+i|k) ||^2_{{\boldsymbol{Q}}(i)} + ||\boldsymbol{\Delta {U}}(k+i|k)||^2_{{\boldsymbol{R}}(i)}" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{\hat{X}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\hat{X}}" title="\boldsymbol{\hat{X}}" /></a> is predicit state by using predict input

- <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{{r}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{{r}}" title="\boldsymbol{{r}}" /></a> is reference state

- <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{\Delta&space;\boldsymbol{U}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\Delta&space;\boldsymbol{U}}" title="\boldsymbol{\Delta \boldsymbol{U}}" /></a> is predict amount of change of input

- <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{\boldsymbol{R}},&space;\boldsymbol{\boldsymbol{Q}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\boldsymbol{R}},&space;\boldsymbol{\boldsymbol{Q}}" title="\boldsymbol{\boldsymbol{R}}, \boldsymbol{\boldsymbol{Q}}" /></a> are evaluation function weights

## Expected Results

- first order system

- time history

<img src = https://github.com/Shunichi09/linear_nonlinear_control/blob/demo_gif/mpc/basic/first_order_states.png width = 70%>

- input

<img src = https://github.com/Shunichi09/linear_nonlinear_control/blob/demo_gif/mpc/basic/first_order_input.png width = 70%>

- ACC (Adaptive cruise control)

- time history of states

<img src = https://github.com/Shunichi09/linear_nonlinear_control/blob/demo_gif/mpc/basic/ACC_states.png width = 70%>

- animation

<img src = https://github.com/Shunichi09/linear_nonlinear_control/blob/demo_gif/mpc/basic/ACC.gif width = 70%>


# Usage

- for example(first order system)

```
$ python main_example.py
```

- for example(ACC (Adaptive cruise control))

```
$ python main_ACC.py
```

- for comparing two methods of optimization solvers

```
$ python test_compare_methods.py
```

# Requirement

- python3.5 or more
- numpy
- matplotlib
- cvxopt
- scipy1.2.0 or more
- python-control

# Reference
I`m sorry that main references are written in Japanese

- モデル予測制御―制約のもとでの最適制御 著：Jan M. Maciejowski　訳：足立修一　東京電機大学出版局