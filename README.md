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
1. Only tested on Ubuntu 20.04

Create a venv:

```bash
python3 -m venv mtl
source mtl/bin/activate
```

Clone repository:

```
git clone ...
cd moo
```

Install requirements:

```
pip install -r requirements.txt
```

Be patient, this takes a while (building wheel for fvcore fails for some reason but is okay anyways).

The large number of dependencies is partly due to the baselines, available in this repository as well.

Finally install the module

```
pip install -e .
```


## Acknowledgments

Many thanks to [submitit](https://github.com/facebookincubator/submitit)!

