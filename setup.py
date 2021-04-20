from setuptools import setup

setup(
    name='cosmos',
    version='0.0.1',
    description='Efficient multi-objective optimization for deep learning',
    url='https://github.com/ruchtem/cosmos',
    author='Michael Ruchte',
    author_email='ruchtem@cs.uni-freiburg.de',
    packages=[
        'multi_objective',
        'plotting'
    ],
    install_requires=[
        "numpy",
    ],
    license='MIT',
    zip_safe=False,
)