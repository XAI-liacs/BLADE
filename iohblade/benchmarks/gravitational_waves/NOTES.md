# Configuration Guide for Runnign this benchmark.

## Using appropriate CUDA.
* The Learn2Design benchmark comes with two dfbench flavours:
  * CUDA 13
  * CUDA 12
* Make sure to use appropriate `cuda_version` when instantiating the problem.

## JAX teathing issues.
* If you are running on multiple GPU setup, dedicate a gpu for jax setup, and make only that GPU visible to JAX by exporting following variable in the environment.
```bash
export CUDA_VISIBLE_DEVICES=<Insert GPU index here>
```
----
