# Enviroments

| Name | Linear | Nonlinear | State Size | Input size |
|:----------|:---------------:|:----------------:|:----------------:|:----------------:|
| First Order Lag System | ✓ | x | 4 | 2 | 
| Two wheeled System (Constant Goal) | x | ✓ | 3 | 2 | 
| Two wheeled System (Moving Goal) (Coming soon) | x | ✓ | 3 | 2 | 
| Cartpole (Swing up) | x | ✓ | 4 | 1 | 

## FistOrderLagEnv

System equations.

<img src="assets/firstorderlag.png" width="550">

You can set arbinatry time constant, tau. The default is 0.63 s

## TwoWheeledEnv

System equations.

<img src="assets/twowheeled.png" width="300">

## CatpoleEnv (Swing up)

System equations.

<img src="assets/cartpole.png" width="600">

You can set arbinatry parameters, mc, mp, l and g. 

Default settings are as follows:

mc = 1, mp = 0.2, l = 0.5, g = 9.8