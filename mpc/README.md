# Model Predictive Control Tool
This program is about template, function of linear model predictive control

# Documentation of this function
Linear model predicitive control should have state equation.
So if you want to use this function, you should model the plant as state equation.
Therefore, the parameters of this class are as following

Parameters :

- A : numpy.ndarray
     - system matrix
- B : numpy.ndarray
    - input matrix
- Q : numpy.ndarray
    - evaluation function weight
- R : numpy.ndarray
    - evaluation function weight
- pre_step : int
    - prediction step
- dt_input_upper : numpy.ndarray
    - constraints of input dt
- dt_input_lower : numpy.ndarray
    - constraints of input dt
- input_upper : numpy.ndarray
    - constraints of input
- input_lower : numpy.ndarray
    - constraints of input

We have two function, mpc_func_with_cvxopt.py and mpc_func_with_scipy.py
Both function have same variable and member function. however the solver is different. 
Plese choose the right method for your environment

## Example
# Problem Formulation and Expected results

- updating soon!!

# Usage

- for example

```
$ python main_example.py
```

- for comparing two methods

```
$ python test_compare_methods.py
```

# Requirement

- python3.5 or more
- numpy
- matplotlib
- cvxopt
- scipy1.2.0 or more

# Reference
I`m sorry that main references are written in Japanese

- モデル予測制御―制約のもとでの最適制御 著：Jan M. Maciejowski　訳：足立修一　東京電機大学出版局