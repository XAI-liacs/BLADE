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
---
