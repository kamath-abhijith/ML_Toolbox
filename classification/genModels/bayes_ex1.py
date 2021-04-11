'''

BAYES CLASSIFICATION WITH GAUSSIAN CLASS-CONDITIONALS

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

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

# path = '/Users/abhijith/Desktop/TECHNOLOGIE/Research/TimeEncodingMachines/Documentation/TEMFRI/TSP/figures/'
# save = True

# %% IMPORT DATA

train_data = np.loadtxt('./data/P1a_train_data_2D.txt', delimiter=',', skiprows=1)
test_data = np.loadtxt('./data/P1a_test_data_2D.txt', delimiter=',', skiprows=1)

# %% TRAINING
pos_mean, neg_mean, pos_cov, neg_cov = \
    bayes_tools.train_gaussian_conditionals(train_data)

# %% PLOTS 
plt.figure(figsize=(12,6))
ax = plt.gca()

bayes_tools.plot_data2D(train_data, ax=ax, xlimits=[-4,10],
    ylimits=[-4,10], show=False)
bayes_tools.plot_boundary([pos_mean, neg_mean], [pos_cov, neg_cov],
    [0.5,0.5], ax=ax, num_points=500, show=False)
bayes_tools.plot_confidence_ellipse2D(pos_mean, pos_cov, nstd=3, ax=ax,
    color='red', show=True)
bayes_tools.plot_confidence_ellipse2D(neg_mean, neg_cov, nstd=3, ax=ax,
    color='green', show=True)

# %% TESTING

confusion_mtx = bayes_tools.test_gaussian_conditionals(test_data,
    [pos_mean, neg_mean], [pos_cov, neg_cov], [0.5, 0.5])

# %% PLOT CONFUSION MATRIX
bayes_tools.plot_confusion_matrix(confusion_mtx)