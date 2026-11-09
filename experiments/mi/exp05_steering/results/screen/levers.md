# Lever screen `screen + screen2`

Lever: concept gain CI above 0 and >= 0.2, failure rate up by at most 0.15. Unsteered failure rate: screen 0.250, screen2 0.125.

| source | direction | dose | concept | concept_rate | d_concept | d_concept_lo | d_concept_hi | failed | d_failed | sim_none | lever |
|---|---|---|---|---|---|---|---|---|---|---|---|
| screen2 | adaptive_parameters21 | 0.100 | adaptive_parameters | 0.000 | -0.125 | -0.312 | 0.000 | 0.062 | -0.062 | 0.782 | False |
| screen2 | adaptive_parameters21 | 0.200 | adaptive_parameters | 0.000 | -0.125 | -0.312 | 0.000 | 0.250 | 0.125 | 0.707 | False |
| screen | adaptive_parameters21 | 0.300 | adaptive_parameters | 0.062 | 0.000 | 0.000 | 0.000 | 0.938 | 0.688 | 0.701 | False |
| screen | adaptive_parameters21 | 0.500 | adaptive_parameters | 0.125 | 0.062 | 0.000 | 0.188 | 1.000 | 0.750 | 0.486 | False |
| screen | adaptive_parameters27 | 0.300 | adaptive_parameters | 0.062 | 0.000 | 0.000 | 0.000 | 0.125 | -0.125 | 0.740 | False |
| screen | adaptive_parameters27 | 0.500 | adaptive_parameters | 0.188 | 0.125 | 0.000 | 0.312 | 0.250 | 0.000 | 0.724 | False |
| screen2 | covariance21 | 0.100 | covariance | 0.188 | 0.062 | 0.000 | 0.188 | 0.250 | 0.125 | 0.764 | False |
| screen2 | covariance21 | 0.200 | covariance | 0.188 | 0.062 | 0.000 | 0.188 | 0.562 | 0.438 | 0.764 | False |
| screen | covariance21 | 0.300 | covariance | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.750 | 0.656 | False |
| screen | covariance21 | 0.500 | covariance | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.750 | 0.206 | False |
| screen | covariance27 | 0.300 | covariance | 0.062 | 0.062 | 0.000 | 0.188 | 0.125 | -0.125 | 0.705 | False |
| screen | covariance27 | 0.500 | covariance | 0.062 | 0.062 | 0.000 | 0.188 | 0.312 | 0.062 | 0.691 | False |
| screen2 | crossover21 | 0.100 | crossover | 0.312 | -0.062 | -0.188 | 0.000 | 0.125 | 0.000 | 0.752 | False |
| screen2 | crossover21 | 0.200 | crossover | 0.312 | -0.062 | -0.188 | 0.000 | 0.375 | 0.250 | 0.710 | False |
| screen | crossover21 | 0.300 | crossover | 0.125 | -0.188 | -0.375 | 0.000 | 1.000 | 0.750 | 0.374 | False |
| screen | crossover21 | 0.500 | crossover | 0.000 | -0.312 | -0.562 | -0.125 | 1.000 | 0.750 | 0.067 | False |
| screen2 | crossover27 | 0.100 | crossover | 0.375 | 0.000 | 0.000 | 0.000 | 0.188 | 0.062 | 0.752 | False |
| screen2 | crossover27 | 0.200 | crossover | 0.375 | 0.000 | -0.188 | 0.188 | 0.062 | -0.062 | 0.734 | False |
| screen | crossover27 | 0.300 | crossover | 0.312 | 0.000 | 0.000 | 0.000 | 0.562 | 0.312 | 0.700 | False |
| screen | crossover27 | 0.500 | crossover | 0.125 | -0.188 | -0.377 | 0.000 | 1.000 | 0.750 | 0.220 | False |
| screen2 | differential_evolution21 | 0.100 | differential_evolution | 0.438 | 0.062 | 0.000 | 0.188 | 0.125 | 0.000 | 0.783 | False |
| screen2 | differential_evolution21 | 0.200 | differential_evolution | 0.438 | 0.062 | 0.000 | 0.188 | 0.312 | 0.188 | 0.738 | False |
| screen | differential_evolution21 | 0.300 | differential_evolution | 0.375 | 0.000 | 0.000 | 0.000 | 0.750 | 0.500 | 0.732 | False |
| screen | differential_evolution21 | 0.500 | differential_evolution | 0.375 | 0.000 | -0.250 | 0.250 | 1.000 | 0.750 | 0.616 | False |
| screen | differential_evolution27 | 0.300 | differential_evolution | 0.250 | -0.125 | -0.312 | 0.000 | 0.125 | -0.125 | 0.689 | False |
| screen | differential_evolution27 | 0.500 | differential_evolution | 0.250 | -0.125 | -0.312 | 0.000 | 0.875 | 0.625 | 0.678 | False |
| screen2 | elitism21 | 0.100 | elitism | 0.062 | 0.000 | -0.188 | 0.188 | 0.188 | 0.062 | 0.802 | False |
| screen2 | elitism21 | 0.200 | elitism | 0.000 | -0.062 | -0.188 | 0.000 | 0.750 | 0.625 | 0.696 | False |
| screen | elitism21 | 0.300 | elitism | 0.188 | -0.062 | -0.312 | 0.188 | 0.750 | 0.500 | 0.631 | False |
| screen | elitism21 | 0.500 | elitism | 0.375 | 0.125 | -0.188 | 0.438 | 1.000 | 0.750 | 0.521 | False |
| screen | elitism27 | 0.300 | elitism | 0.125 | -0.125 | -0.312 | 0.000 | 0.125 | -0.125 | 0.687 | False |
| screen | elitism27 | 0.500 | elitism | 0.188 | -0.062 | -0.252 | 0.125 | 0.312 | 0.062 | 0.650 | False |
| screen2 | gradient21 | 0.100 | gradient | 0.750 | 0.125 | -0.125 | 0.375 | 0.312 | 0.188 | 0.721 | False |
| screen2 | gradient21 | 0.200 | gradient | 0.812 | 0.188 | 0.000 | 0.375 | 0.000 | -0.125 | 0.721 | False |
| screen | gradient21 | 0.300 | gradient | 0.812 | 0.250 | 0.062 | 0.500 | 0.062 | -0.188 | 0.665 | True |
| screen | gradient21 | 0.500 | gradient | 0.500 | -0.062 | -0.438 | 0.312 | 0.500 | 0.250 | 0.317 | False |
| screen | gradient27 | 0.300 | gradient | 0.812 | 0.250 | 0.062 | 0.500 | 0.062 | -0.188 | 0.687 | True |
| screen | gradient27 | 0.500 | gradient | 0.875 | 0.312 | 0.125 | 0.562 | 0.125 | -0.125 | 0.670 | True |
| screen2 | local_search21 | 0.100 | local_search | 0.062 | 0.062 | 0.000 | 0.188 | 0.125 | 0.000 | 0.722 | False |
| screen2 | local_search21 | 0.200 | local_search | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.125 | 0.740 | False |
| screen | local_search21 | 0.300 | local_search | 0.125 | 0.062 | -0.125 | 0.250 | 0.938 | 0.688 | 0.689 | False |
| screen | local_search21 | 0.500 | local_search | 0.250 | 0.188 | 0.000 | 0.375 | 1.000 | 0.750 | 0.238 | False |
| screen | local_search27 | 0.300 | local_search | 0.000 | -0.062 | -0.188 | 0.000 | 0.062 | -0.188 | 0.727 | False |
| screen | local_search27 | 0.500 | local_search | 0.000 | -0.062 | -0.188 | 0.000 | 0.188 | -0.062 | 0.695 | False |
| screen2 | particle_swarm21 | 0.100 | particle_swarm | 0.250 | 0.125 | 0.000 | 0.312 | 0.125 | 0.000 | 0.776 | False |
| screen2 | particle_swarm21 | 0.200 | particle_swarm | 0.188 | 0.062 | -0.250 | 0.375 | 0.312 | 0.188 | 0.696 | False |
| screen | particle_swarm21 | 0.300 | particle_swarm | 0.938 | 0.812 | 0.625 | 1.000 | 0.812 | 0.562 | 0.702 | False |
| screen | particle_swarm21 | 0.500 | particle_swarm | 0.438 | 0.312 | 0.000 | 0.625 | 1.000 | 0.750 | 0.282 | False |
| screen | particle_swarm27 | 0.300 | particle_swarm | 0.375 | 0.250 | 0.000 | 0.500 | 0.375 | 0.125 | 0.722 | False |
| screen | particle_swarm27 | 0.500 | particle_swarm | 0.938 | 0.812 | 0.625 | 1.000 | 0.938 | 0.688 | 0.637 | False |
| screen | swarm | 0.300 | particle_swarm | 0.938 | 0.812 | 0.625 | 1.000 | 0.188 | -0.062 | 0.681 | True |
| screen | swarm | 0.500 | particle_swarm | 1.000 | 0.875 | 0.688 | 1.000 | 0.062 | -0.188 | 0.634 | True |
| screen2 | restart21 | 0.100 | restart | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.795 | False |
| screen2 | restart21 | 0.200 | restart | 0.000 | 0.000 | 0.000 | 0.000 | 0.188 | 0.062 | 0.753 | False |
| screen | restart21 | 0.300 | restart | 0.000 | -0.062 | -0.188 | 0.000 | 0.125 | -0.125 | 0.730 | False |
| screen | restart21 | 0.500 | restart | 0.500 | 0.438 | 0.125 | 0.688 | 0.875 | 0.625 | 0.629 | False |
| screen | restart27 | 0.300 | restart | 0.000 | -0.062 | -0.188 | 0.000 | 0.188 | -0.062 | 0.680 | False |
| screen | restart27 | 0.500 | restart | 0.000 | -0.062 | -0.188 | 0.000 | 0.375 | 0.125 | 0.663 | False |
| screen2 | surrogate21 | 0.100 | surrogate | 0.062 | 0.062 | 0.000 | 0.188 | 0.125 | 0.000 | 0.711 | False |
| screen2 | surrogate21 | 0.200 | surrogate | 0.125 | 0.125 | 0.000 | 0.312 | 0.312 | 0.188 | 0.652 | False |
| screen | surrogate21 | 0.300 | surrogate | 0.062 | 0.062 | 0.000 | 0.188 | 0.438 | 0.188 | 0.641 | False |
| screen | surrogate21 | 0.500 | surrogate | 0.125 | 0.125 | 0.000 | 0.312 | 0.688 | 0.438 | 0.512 | False |
| screen2 | surrogate27 | 0.100 | surrogate | 0.062 | 0.062 | 0.000 | 0.188 | 0.062 | -0.062 | 0.721 | False |
| screen2 | surrogate27 | 0.200 | surrogate | 0.188 | 0.188 | 0.000 | 0.375 | 0.500 | 0.375 | 0.583 | False |
| screen | surrogate27 | 0.300 | surrogate | 0.250 | 0.250 | 0.062 | 0.438 | 0.812 | 0.562 | 0.490 | False |
| screen | surrogate27 | 0.500 | surrogate | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.241 | False |

## Selected (2)

- `swarm` (L27 f38837) at dose 0.5: particle_swarm +0.88, failures 0.06
- `gradient27` (L27 f49680) at dose 0.5: gradient +0.31, failures 0.12
