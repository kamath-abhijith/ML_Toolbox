'''

BAYES CLASSIFICATION USING GAUSSIAN
MIXTURE MODEL ESTIMATION

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

import bayes_tools

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 11})

os.makedirs('./results', exist_ok=True)
path = './results/'

# %% IMPORT DATA

train_data = np.loadtxt('./data/P1c_train_data_2D.txt', delimiter=',', skiprows=1)
test_data = np.loadtxt('./data/P1c_test_data_2D.txt', delimiter=',', skiprows=1)

# %% TRAIN GMM

priors, means, covs, cost = bayes_tools.train_gmm(train_data, num_components=2,
    max_iter=200, tol=1e-12)

# %% PLOT
fig = plt.figure(figsize=(12,6))
ax = plt.gca()
bayes_tools.plot_loss(cost, ax=ax, yaxis_label=r'$\ln f(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$')

fig = plt.figure(figsize=(12,6))
ax = plt.gca()
bayes_tools.plot_data2D(train_data, ax=ax, xlimits=[-4,10],
    ylimits=[-4,10], show=False)
bayes_tools.plot_confidence_ellipse2D(means[0], covs[0], nstd=3, ax=ax,
    color='red', show=True)
bayes_tools.plot_confidence_ellipse2D(means[1], covs[1], nstd=3, ax=ax,
    color='green', show=True)
# %%
