AlphaEvolve Benchmarks
======================

BLADE includes benchmark instances inspired by the Google DeepMind
AlphaEvolve paper. These instances are available in two complementary forms:

- ``run_benchmarks/`` provides standalone reference scripts for running each
  task directly.
- ``iohblade/benchmarks`` packages the same tasks for programmatic use in
  experiments and pipelines.

The packaged benchmarks are grouped by domain:

- Analysis (auto-correlation inequalities)
- Combinatorics (Erdos min-overlap)
- Geometry (Heilbronn problems, kissing number, and distance ratios)
- Matrix multiplication
- Number theory (sums vs differences)
- Packing (rectangle, hexagon, and unit square packing)
- Fourier (uncertainty inequalities)

Each domain folder contains a README with task-specific details and citations
to the original sources.
