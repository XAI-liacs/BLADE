# Cutscenes

Stills (1920×1080) for the tutorial videos, in the style of the intro and
outro (`../intro-outro`). Hold each for 2–5 s, or use as the first/last
frame of a short Veo clip: [`VEO.md`](VEO.md) has a prompt for every
scene.

| Frame | Content |
|-------|---------|
| `00-blank` | only the background, for your own titles or text |
| `bench-01-bbob` | `bbob`: all 24 BBOB functions (real 2-D landscapes) |
| `bench-02-bbob-groups` | `bbob_groups`: one function per group (f3, f8, f11, f16, f22) |
| `bench-03-ma-bbob` | `ma_bbob`: a many-affine landscape (instance 21) |
| `bench-04` … `bench-15` | packing, point configurations, spheres, analysis, sums vs differences, matrix multiplication, trees and colourings, routing, AutoML, kernel tuning, photonics, multi-objective |
| `ceg` | code evolution graph (illustrative data) |
| `convergence` | convergence curves, mean and spread over seeds (illustrative, methods A–D) |
| `framework` | how BLADE works, five stages |
| `framework-detailed` | the platform figure (`docs/media/blade-platform-dark.png`) |
| `aad-loop` | task prompt → LLM writes code → population → scoring → selection → LLM mutation |
| `methods` | the nine built-in methods |
| `isolation` | generated code → sandbox (process, environment, time limit) → score, error or timeout |
| `llms` | hosted and local LLM providers, `provider: multi` |
| `yaml-terminal` | the episode 2 experiment next to `validate`, `check` and `run` |
| `episode-1` … `episode-7` | title cards |
| `next-1` … `next-6` | "up next" cards for the end of episodes 1–6 |
| `lower-third` | name and role, transparent background, to overlay on the camera shot |

The landscapes are computed with `ioh` (2-D, log-scaled, colours by rank:
amber is near the optimum, the white dot marks it).

## Rendering

The frames are not committed (about 40 MB); generate them with:

```bash
uv run python docs/videos/cutscenes/make_cutscenes.py          # all, or name prefixes
NODE_PATH=$(npm root -g) node docs/videos/cutscenes/render.js  # SVG -> PNG
```

`make_cutscenes.py` takes name prefixes to rebuild only some frames, for
example `make_cutscenes.py bench-03 aad`, and so does `render.js`.
