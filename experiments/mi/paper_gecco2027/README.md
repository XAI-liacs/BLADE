# GECCO 2027 paper draft (L4EC track)

Outline draft of a full paper for the *Learning for Evolutionary
Computation* track of GECCO 2027. Abstract (200 words) due 18 Jan 2027,
paper due 25 Jan 2027. Limit: 8 pages without references; double-blind.

* `main.tex`: ACM `acmart` with the class options from the GECCO 2027
  call for papers:
  `\documentclass[dvipsnames, format=sigconf, anonymous=true, review=true]{acmart}`.
* `sections/`: one file per section. Each section lists the planned
  content (blue), the supporting numbers, their source in
  `experiments/mi` (grey) and a page budget; `\todo{}` (red) marks open
  items.
* `refs.bib`: entries marked `VERIFY` were written from memory and must
  be checked against the publisher.
* `PLAN.md`: experiments to complete the story before the deadline.

## Build

`acmart.cls` and `ACM-Reference-Format.bst` ship with TeX Live and
Overleaf, so they are not vendored here. Use the official ACM package
from CTAN or the ACM template page.

```bash
latexmk -pdf main.tex
```

On Overleaf: upload this folder (or start from the ACM "Primary Article
Template" and replace `main.tex`).

The draft has not been compiled yet (no TeX installation on the
experiment machine).

## Open decisions

* Title. Alternatives: "Reading and Steering the Generator in LLM-driven
  Algorithm Design"; "What an LLM Knows While It Designs Optimisers".
* Whether the counterfactual analysis (C2) stays in the main text.
* Whether the benchmark fixes are a contribution or a setup paragraph.
* The CCS concepts (check the ids with the ACM CCS tool).
