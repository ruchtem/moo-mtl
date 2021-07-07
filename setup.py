from setuptools import setup

setup(
    name='moo',
    version='0.0.1',
    description='Multi-Task problems are not multi-objective',
    url='',
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