# The 2025 Imsil earthquake swarm: summary report (14 October 2026)

**`report.pdf`** is an 11-page summary of the project, written mainly as methods and results:
- detection and location;
- relocation and the 25 m reference events;
- magnitudes;
- template matching and geometry;
- migration;
- comparison with Park et al. (2026);
- the long template-matching scan;
- the events of 15 April 2019.

**`figures/`** holds the seven figures as `Figure_NN_<topic>.pdf` and `.png` (300 dpi), and `sources.json` lists the input files of each figure with checksums.

## How every number gets into the text

No number in `report.tex` is typed by hand.
- Notebook `notebooks/16_report_figures.ipynb` writes `facts.json` (value and source of each number) and `facts.tex` (one macro `\F{key}` per number).
- A fact is either computed in notebook 16 from the final catalogs, or quoted from the executed output of the notebook that measured it, with a pattern that must match exactly one value.
- Values that both notebook 16 and the measuring notebook compute are asserted to be equal.
- An undefined key stops the LaTeX run.

## Rebuild

    ~/miniforge3/envs/dl-base/bin/python notebooks/build_16_report_figures.py
    jupyter execute --inplace --kernel_name=dl-base notebooks/16_report_figures.ipynb    # ~6 min; figures/, facts.json, facts.tex
    tectonic -X compile report.tex                                                     # in this folder

`refs.bib` copies its entries from the Haenam summary report, plus Park et al. (2026) from the paper and Sheen et al. (2018) from the reference list of Park et al. (2026).
