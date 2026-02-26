# Packing Benchmarks

This directory contains implementations of circle packing optimization benchmarks based on the AlphaEvolve paper.

## Benchmarks Implemented

### B.7: Unit Regular Hexagons in Regualar Hexagon (n=11,12)
* Pack $n$ disjoint regular hexagons of side 1 inside a larger regular hexagon, minimizing the outer side length.
    * $n \in {11, 12}$
* We verify disjoint interiors and compute the minimal required outer side length $L$ via support functions.
* The score to minimise is $L$.

### B.12: Unit Square Circle Packing
- **Problem**: Pack n disjoint circles inside a unit square $[0,1] × [0,1]$ to maximize the sum of their radii
- **Instances**: 
  - n=26 circles
  - n=32 circles

### B.13: Rectangle Circle Packing  
- **Problem**: Pack n disjoint circles inside a rectangle with perimeter 4 to maximize the sum of their radii
- **Instance**: n=21 circles with perimeter constraint of 4

### Circle Packing
- **Problem**: Pack $n$ disjoint circles inside a given circle.
- **Instance**: Given a container circle $C_0$, and a set of $n$ candidate circles $c \in C \quad |C| = n$, with radii $R_i\quad i \in 1\{1,\ldots, n\}$ and a tolerance factor $\epsilon$.
  - Optimisation: 
    * Generate a decision vector $\alpha = |\alpha_1, \ldots, \alpha_n|$:
    $$
      \alpha_i = \begin{cases}
      0 & \text{circle $c_i$ is not selected.}\\
      1 & \text{circle $c_i$ is selected.}\\
      \end{cases}
    $$
    * Such that the sum of area of those circles is maximised:
      $$O = \max_{i = 1}^n \alpha_i \pi R_i^2$$
    * None of the included circles overlap each other, within the tolerance:
    $$
      dist(c_i, c_j) \ge r_i + r_j + \epsilon \quad \forall i, j; i\ne j; \alpha_i; \alpha_j = 1
    $$
    * None of the included circles overlap the container, within the tolerance:
    $$
      dist(C_0, c_i) \ge r_0 + r_i + \epsilon \quad \forall i; \alpha_i = 1
    $$
---
