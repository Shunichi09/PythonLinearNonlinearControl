# Model Predictive Control for Vehicle model
This program is for controlling the vehicle model.
I implemented the steering control for vehicle by using Model Predictive Control.

# Model
Usually, the vehicle model is expressed by extremely complicated nonlinear equation.
Acoording to reference 1, I used the simple model as shown in following equation.

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;x[k&plus;1]&space;=&space;x[k]&space;&plus;&space;v\cos(\theta[k])dt&space;\\&space;y[k&plus;1]&space;=&space;y[k]&space;&plus;&space;v\sin(\theta[k])dt&space;\\&space;\theta[k&plus;1]&space;=&space;\theta[k]&space;&plus;&space;\frac{v&space;\tan{\delta[k]}}{L}dt\\&space;\delta[k&plus;1]&space;=&space;\delta[k]&space;-&space;\tau^{-1}(\delta[k]-\delta_{input})dt&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;x[k&plus;1]&space;=&space;x[k]&space;&plus;&space;v\cos(\theta[k])dt&space;\\&space;y[k&plus;1]&space;=&space;y[k]&space;&plus;&space;v\sin(\theta[k])dt&space;\\&space;\theta[k&plus;1]&space;=&space;\theta[k]&space;&plus;&space;\frac{v&space;\tan{\delta[k]}}{L}dt\\&space;\delta[k&plus;1]&space;=&space;\delta[k]&space;-&space;\tau^{-1}(\delta[k]-\delta_{input})dt&space;\end{align*}" title="\begin{align*} x[k+1] = x[k] + v\cos(\theta[k])dt \\ y[k+1] = y[k] + v\sin(\theta[k])dt \\ \theta[k+1] = \theta[k] + \frac{v \tan{\delta[k]}}{L}dt\\ \delta[k+1] = \delta[k] - \tau^{-1}(\delta[k]-\delta_{input})dt \end{align*}" /></a>

However, it is still a nonlinear equation.
Therefore, I assume that the car is tracking the reference trajectory.
If we get the assumption, the model can turn to linear model by using the path's curvatures.

<a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{X}[k&plus;1]&space;=&space;\begin{bmatrix}&space;1&space;&&space;vdt&space;&&space;0&space;\\&space;0&space;&&space;1&space;&&space;\frac{vdt}{L&space;\&space;cos^{2}&space;\delta_r}&space;\\&space;0&space;&&space;0&space;&&space;1&space;-&space;\tau^{-1}dt\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;y[k]&space;\\&space;\theta[k]&space;\\&space;\delta[k]&space;\end{bmatrix}&space;&plus;&space;\begin{bmatrix}&space;0&space;\\&space;0&space;\\&space;\tau^{-1}dt&space;\\&space;\end{bmatrix}&space;\delta_{input}&space;-&space;\begin{bmatrix}&space;0&space;\\&space;-\frac{v&space;\delta_r&space;dt}{L&space;\&space;cos^{2}&space;\delta_r}&space;\\&space;0&space;\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{X}[k&plus;1]&space;=&space;\begin{bmatrix}&space;1&space;&&space;vdt&space;&&space;0&space;\\&space;0&space;&&space;1&space;&&space;\frac{vdt}{L&space;\&space;cos^{2}&space;\delta_r}&space;\\&space;0&space;&&space;0&space;&&space;1&space;-&space;\tau^{-1}dt\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;y[k]&space;\\&space;\theta[k]&space;\\&space;\delta[k]&space;\end{bmatrix}&space;&plus;&space;\begin{bmatrix}&space;0&space;\\&space;0&space;\\&space;\tau^{-1}dt&space;\\&space;\end{bmatrix}&space;\delta_{input}&space;-&space;\begin{bmatrix}&space;0&space;\\&space;-\frac{v&space;\delta_r&space;dt}{L&space;\&space;cos^{2}&space;\delta_r}&space;\\&space;0&space;\\&space;\end{bmatrix}" title="\boldsymbol{X}[k+1] = \begin{bmatrix} 1 & vdt & 0 \\ 0 & 1 & \frac{vdt}{L \ cos^{2} \delta_r} \\ 0 & 0 & 1 - \tau^{-1}dt\\ \end{bmatrix} \begin{bmatrix} y[k] \\ \theta[k] \\ \delta[k] \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ \tau^{-1}dt \\ \end{bmatrix} \delta_{input} - \begin{bmatrix} 0 \\ -\frac{v \delta_r dt}{L \ cos^{2} \delta_r} \\ 0 \\ \end{bmatrix}" /></a>

and \delta_r denoted

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_r&space;=&space;\arctan(\frac{L}{R})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_r&space;=&space;\arctan(\frac{L}{R})" title="\delta_r = \arctan(\frac{L}{R})" /></a>

R is the curvatures of the reference trajectory.

Now we can get the linear state equation and can apply the MPC theory.

However, you should care that this state euation could be changed during the predict horizon.
I implemented this, so if you know about the detail please go to "IteraticeMPC_func.py"

# Expected Results

<img src = https://github.com/Shunichi09/linear_nonlinear_control/blob/demo_gif/mpc/extend/animation_all.gif width = 70%>

<img src = https://github.com/Shunichi09/linear_nonlinear_control/blob/demo_gif/mpc/extend/animation_zoom.gif width = 70%>

# Usage

```
$ python main_track.py 
```

# Reference
- 1. https://qiita.com/taka_horibe/items/47f86e02e2db83b0c570#%E8%BB%8A%E4%B8%A1%E3%81%AE%E8%BB%8C%E9%81%93%E8%BF%BD%E5%BE%93%E5%95%8F%E9%A1%8C%E9%9D%9E%E7%B7%9A%E5%BD%A2%E3%81%AB%E9%81%A9%E7%94%A8%E3%81%99%E3%82%8B (Japanese)
