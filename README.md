# Multi-Task problems are not multi-objective

This is the code for the paper "Multi-Task problems are not multi-objective" in which we show
that the commonly used Multi-Fashion-MNIST datasets are not suitable for benchmarking multi-objective
methods.

For more details see the paper.

## Usage

```python
python multi_objective/main.py --config path/to/config.yaml
```

Config files can be found in [configs](configs).

There is also the option to set options using the command line:

```python
python multi_objective/main.py epochs 100
```

For reproducing the results of the paper see the jupyter notebooks [generate_results](generate_results.ipynb). For the HPO see [hpo](hpo.ipynb).

## Installation

Requirements:
1. Only tested on Ubuntu 20.04.
1. `python >= 3.7`

Create a venv:

```bash
python3 -m venv mtl
source mtl/bin/activate
```

Clone repository:

```
git clone https://github.com/ruchtem/moo-mtl.git
cd moo-mtl
```

Upgrade pip and install requirements:

```
pip install --upgrade pip
pip install -r requirements.txt
```

Be patient, this takes a while.

The large number of dependencies is partly due to the baselines, available in this repository as well. If `cvxopt` or `cvxpy` give you trouble (e.g. `ERROR: Failed building wheel for scs`) you can omit them, they are only required for the EPO part of PHN.

Finally install the module in editable mode

```
pip install -e .
```


## Acknowledgments

I would like to thank [Samuel Müller](https://github.com/SamuelGabriel) for many helpful discussions and suggestions.

Many thanks also to [submitit](https://github.com/facebookincubator/submitit)!

