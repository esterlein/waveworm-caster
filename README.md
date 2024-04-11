# **WAVEWORM-FIELD-CASTER**

## About
This repository is part of the [project](https://github.com/esterlein/stm32-waveworm) to visualize electromagnetic fields in three-dimensional space.
The field is simulated by a matrix filled with several topological patterns. At the current stage of research and development, a simplified version of the topology is used: two matrices in orthogonal planes fully describe the electromagnetic field enclosed by the surface.
A random function is used to get the probing matrices from field matrices, but since it is difficult to obtain a uniform distribution on a sample size being researched, depth-first search with backtracking is utilized.
Thus, the problem of topological field interpolation is equivalent to the matrix completion problem, and several adaptive methods can be implemented to solve the problem with an acceptable error level.

## Example
```
Size: 20
Strength: 12 (assumed range from ADC)
Field type: Radial Gradient
Noise level: 2
Probing density: 10 (% of matrix size)
Probing sparsity: 2 (depth in backtracking DFS)
```

Nuclear norm minimization implemented in [fancyimpute](https://github.com/iskandr/fancyimpute):

![alt text](https://waveworm.io/img/NNM_D10_S2_N2.png)
