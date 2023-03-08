[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-1.14.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
# Variational Autoencoder for gene expression data
Implemented using PyTorch for Statistical Data Analysis 2 class at University of Warsaw.

Original paper about VAE:

Kingma, D. P. & Welling, M. (2014), Auto-Encoding Variational Bayes, in '2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings' .
# Project overview
Project overview can be seen in `project_overview.pdf`.

# Dataset
In this project we use data from single cell RNA sequencing.
The dataset was released as a part of
[NeurIPS 2021 ”Open problems in Single Cell Analysis” competition](https://openproblems.bio/neurips_2021/).

# How to run
## Installing dependencies
1. Create the environment from the `environment.yml` file:

    `conda env create -f environment.yml`

2. Activate the new environment:

    `conda activate myenv`

## Training model
In order to train model run `python3 train.py`.
Trained models are saved in the `Models` directory.

## Making plots
In order to make plots run `python3 eval.py` (after the training is complete).
Plots are saved in the `Plots` directory.

# Repository structure
## Main directory
1. `report.pdf` contains report from exploring the dataset
and experiments involving the variational autoencoder.
2. `report.tex` is a LaTeX code used for compiling report in pdf.
2. `train.py` is a script used for training the VAE.
3. `eval.py` is a script which uses trained models and makes various plots.

## Variational Autoencoder
VAE class and its dependencies are stored in the src directory.
1. `VAE.py` contains VAE class.
2. `Encoder.py` contains encoder class used in VAE.
3. `Decoder.py` contains simple class decoder used in VAE. Uses gaussian distribution.
4. `CustomDecoder.py` contains custom decoder class designed for this task, which uses Poisson distribution.



