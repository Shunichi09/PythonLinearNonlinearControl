# Model Predictive Control Tool
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

** updating soon !!

- first order system


- ACC (Adaptive cruise control)



## Expected Results

- first order system


- ACC (Adaptive cruise control)


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