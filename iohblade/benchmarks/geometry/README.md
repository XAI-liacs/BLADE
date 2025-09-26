# Geometry benchmarks

This folder implements the geometry problems from the paper.

## Problems and scoring
* All evaluators use hard feasibility checks.

## Heilbronn on a unit-area triangle (n=11).
* Find ($n$) points on or inside a triangle of area 1 that maximize the minimum area of any triangle formed by the points.
    * Our evaluator accepts either:
        * `Points` alone and scales them to fit inside a predefined default unit-area triangle.
        * `(triangle, points)` and rescales the triangle to area 1. 
        * `{"triangle" : triangle, "points": [point]}`: A dictionary with key, "triangle" and "points", both are scaled similarly to make the triangle of unit area.
    * $Score = (\min_{a,b,c} \max\sqrt{s (s-a)(s-b)(s-c)})$.
        * Where $a,b,c \in \texttt{points}$.
            * $a \ne b$
            * $b \ne c$
            * $a \ne c$
            * $s = \frac{a+b+c}2$
            * $\max\sqrt{s (s-a)(s-b)(s-c)}$ is largest area of triangle abc, formed by points.


## Heilbronn on a unit-area convex region (n=13,14).
* Find ($n$) points in a convex region of area 1 that maximize the minimum triangle area.
* We take the convex hull of the points as the region and rescale to area $1$
* Score = minimum triangle area.
    * Scored for points $n\in\{13, 14\}$


## Min/max distance ratio (2D n=16, 3D n=14).
* Minimize $(\frac{\max_{i<j} d(i,j) }{ \min_{i<j} d(i,j)})^2$.
* We have two benchmarks here:
    * For 2D space, number of points $n = 16$.
    * For 3D space, number of points $n = 14$.

## Kissing number in 11D.
* Return an integer set $C\subset\mathbb{Z}^{11}\setminus{0}$ with constraint $\min_{x\ne y}|x-y|\ge \max_x|x|$. 
    * By Lemma 1, this implies a kissing configuration of size (|C|).
* Score $=|C|$.

## Notes on alignment
* Triangles/convex regions are unit-area by construction (we rescale if needed), matching the paper’s normalization.
* Hexagon packing uses disjoint interiors and reports the outer side length; we compute it from apothem ranges with $L=\frac{2a}{\sqrt{3}}$.
* Distance-ratio results are reported as the squared ratio; our objective matches that quantity.
* For 11D kissing numbers we enforce integer coordinates and the lemma condition.
---