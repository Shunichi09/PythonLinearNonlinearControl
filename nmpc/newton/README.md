# Newton method of Nonlinear Model Predictive Control
This program is about NMPC with newton method.
Usually we have to calculate the partial differential of optimal matrix.
In this program, in stead of using any paticular methods to calculate the partial differential of optimal matrix, I used numerical differentiation.
Therefore, I believe that it easy to understand and extend your model.

# Problem Formulation

- **example**

- model

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;\dot{x_1}&space;\\&space;\dot{x_2}&space;\\&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;x_2&space;\\&space;(1-x_1^2-x_2^2)x_2-x_1&plus;u&space;\\&space;\end{bmatrix},&space;|u|&space;\leq&space;0.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;\dot{x_1}&space;\\&space;\dot{x_2}&space;\\&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;x_2&space;\\&space;(1-x_1^2-x_2^2)x_2-x_1&plus;u&space;\\&space;\end{bmatrix},&space;|u|&space;\leq&space;0.5" title="\begin{bmatrix} \dot{x_1} \\ \dot{x_2} \\ \end{bmatrix} = \begin{bmatrix} x_2 \\ (1-x_1^2-x_2^2)x_2-x_1+u \\ \end{bmatrix}, |u| \leq 0.5" /></a>

- evaluation function

To consider the constraints of input u, I introduced dummy input.

<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;\frac{1}{2}(x_1^2(t&plus;T)&plus;x_2^2(t&plus;T))&plus;\int_{t}^{t&plus;T}\frac{1}{2}(x_1^2&plus;x_2^2&plus;u^2)-0.01vd\tau" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J&space;=&space;\frac{1}{2}(x_1^2(t&plus;T)&plus;x_2^2(t&plus;T))&plus;\int_{t}^{t&plus;T}\frac{1}{2}(x_1^2&plus;x_2^2&plus;u^2)-0.01vd\tau" title="J = \frac{1}{2}(x_1^2(t+T)+x_2^2(t+T))+\int_{t}^{t+T}\frac{1}{2}(x_1^2+x_2^2+u^2)-0.01vd\tau" /></a>


- **two wheeled model**

coming soon !

# Expected Results

- example

![Figure_1.png]()

you can confirm that the my method could consider the constraints of input.

- two wheeled model

coming soon !

# Usage

- for example

```
$ python main_example.py
```

- for two wheeled

coming soon !

# Requirement

- python3.5 or more
- numpy
- matplotlib

# Reference
I`m sorry that main references are written in Japanese

- main (commentary article) (Japanse)　https://qiita.com/MENDY/items/4108190a579395053924

- Ohtsuka, T., & Fujii, H. A. (1997). Real-time Optimization Algorithm for Nonlinear Receding-horizon Control. Automatica, 33(6), 1147–1154. https://doi.org/10.1016/S0005-1098(97)00005-8

- 非線形最適制御入門（コロナ社）

- 実時間最適化による制御の実応用（コロナ社）