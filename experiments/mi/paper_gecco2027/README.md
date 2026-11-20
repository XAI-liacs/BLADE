# GECCO 2027 paper draft (L4EC track)

Prose draft of a full paper for the *Learning for Evolutionary
Computation* track of GECCO 2027. Abstract (200 words) due 18 Jan 2027,
paper due 25 Jan 2027. Limit: 8 pages without references; double-blind.

* `main.tex`: ACM `acmart` with the class options from the GECCO 2027
  call for papers:
  `\documentclass[dvipsnames, format=sigconf, anonymous=true, review=true]{acmart}`.
* `sections/`: one file per section; `\todo{}` (red) marks open items.
  `09_appendix.tex` holds the appendices (compiled after the
  references; for submission they become the supplementary material).
* `sections/gen/`: appendix tables, generated from the committed
  results by `figures/make_appendix.py`. Do not edit by hand.
* `refs.bib`: entries marked `VERIFY` were written from memory and must
  be checked against the publisher.
* `PLAN.md`: experiments to complete the story before the deadline.

## Build

`acmart.cls` and `ACM-Reference-Format.bst` ship with TeX Live and
Overleaf, so they are not vendored here. Figures are made from the
committed exp05 results:

```bash
python figures/make_figures.py    # dial.pdf, budget200.pdf
python figures/make_appendix.py   # probes*.pdf, curves_transfer.pdf, sections/gen/*.tex
latexmk -pdf main.tex
```

On the experiment machine (TeX Live 2026 in `/usr/local/texlive/2026`)
`latexmk` and the format build fail because the system Perl lacks
`File::Copy`/`File::Find`. Build the format once without Perl and run
pdflatex/bibtex directly:

```bash
export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH
export TEXFORMATS=$MI_CACHE/texfmt:
(mkdir -p $MI_CACHE/texfmt && cd $MI_CACHE/texfmt && pdftex -ini -jobname=pdflatex -progname=pdflatex -etex pdflatex.ini)
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Last compiled 2026-11-19: main text 5.6 pages, references to page 6, appendix pages 7-12.
overfull lines. The `.gitignore` keeps `main.pdf` and the LaTeX aux
files out of git.

On Overleaf: upload this folder (or start from the ACM "Primary Article
Template" and replace `main.tex`).

## Open decisions

* Title. Working title "Opening the Black-Box Operator: Mechanistic
  Interpretability for LLM-Driven Algorithm Design" (story: what the
  operator knows, what it listens to, how to control it). Decide after
  the confirmation: if it holds, "A Step Size for the LLM: Mechanistic
  Control of Variation in Automated Algorithm Design" is an option.
  Earlier: "What Does the Generator Know? Reading and Steering ...".
* Whether the benchmark fixes are a contribution or a setup paragraph.
* The CCS concepts (check the ids with the ACM CCS tool).
