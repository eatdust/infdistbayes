# Infdistbayes: analysis tools for Bayaspic

---

### Summary

Infdistbayes is a set of python programs and modules to analyse the
massively parallel output of
[bayaspic](https://github.com/cosmicinflation/bayaspic). They are
based on [GetDist](https://pypi.org/project/getdist/),
[Anesthetic](https://pypi.org/project/anesthetic/),
[NumPy](https://pypi.org/project/numpy/),
[SciPy](https://pypi.org/project/scipy/) and
[Matplotlib](https://pypi.org/project/matplotlib/) that should be
installed on your system.

---

### Usage

A [bash](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) script,
**getbayes.bash**, call the various python programs to analyse the
chains output by Bayaspic. These ones should be present within the
"chains/" directory. The bash script contains various editable options
requiring edition before a run.

Notice that some output requires statistical data computed over the
prior space, i.e., derived from
[bayaspic](https://github.com/cosmicinflation/bayaspic) with a flat
likelihood. In that situation, the bootstrapping consists in running
[infdistbayes]() over the pure-prior chains and move all output files
from "data/" to "null/". Then, the production run can start and will
use the prior statistics available within the directory "null/".

---

### Output

Infdistbayes outputs, for all models, various logs in the directory
"output/", statistics within "stats/", marginalized posteriors within
"data/" as well as 1D and 2D contour plots within "plots/". At the end
of the run, the top directory contains a text file "bayesdist.log"
listing model names, evidences, number of parameters and other
meaningfull statistics in model space.

The directory src/ also contains various python programs to plot these
statistics. They require the "bayesdist.log" file to be present
(check-out their command-line help for more information).

See [infdistbayes-output]() for an example.

