'''

TOOLS FOR PARAMETRIC BAYES CLASSIFICATION

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% IMPORT DATA

import numpy as np
import seaborn as sns

from tqdm import tqdm

from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

# %% PLOTTING

def plot_data2D(data, ax=None, title_text=None,
    xlimits=[-4,10], ylimits=[-4,10], show=True, save=False):
    ''' Plots 2D data with labels '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    labels = data[:,2]
    pos_samples = data[np.where(labels==1)][:,:2]
    neg_samples = data[np.where(labels==-1)][:,:2]

    plt.scatter(pos_samples[:,0], pos_samples[:,1], color='red')
    plt.scatter(neg_samples[:,0], neg_samples[:,1], color='green')

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title_text)

    plt.ylim(ylimits)
    plt.xlim(xlimits)

    if show:
        plt.show()
        if save:
            plt.savefig(save + '.pdf', format='pdf')

    return

def plot_confidence_ellipse2D(mean, cov, nstd=3, ax=None, color='black', 
    line_width=2, show=True):
    ''' Plots confidence ellipse for Gaussian data '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    height, width = nstd*np.sqrt(np.linalg.eig(cov)[0])
    ellipse = Ellipse(xy=(mean[0], mean[1]), width=width, height=height,
        angle=-180.0*np.arctan((width-cov[0,0])/cov[1,0])/np.pi, linewidth=line_width,
        facecolor='blue', alpha=0.2, edgecolor=color)
    
    if show:
        ax.add_patch(ellipse)

    return

def plot_confusion_matrix(data, ax=None, xaxis_label=r'PREDICTED CLASS',
    yaxis_label=r'TRUE CLASS', show=True, save=False):
    ''' Plots confusion matrix '''
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()

    ax = sns.heatmap(data, vmin=0.0, vmax=1.0, linewidths=0.5, annot=True)
    # ax.invert_yaxis()
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    if show:
        if save:
            plt.savefig(save + '.pdf', format='pdf')
        plt.show()

    return

def plot_loss(loss, ax=None, xaxis_label=r'NUMBER OF ITERATIONS',
    yaxis_label=r'LOSS', colour='blue', line_width=4,
    show=True, save=False):
    ''' Plots loss function vs iterations '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(loss, color=colour, linewidth=line_width)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    if show:
        if save:
            plt.savefig(save + '.pdf', format='pdf')
        plt.show()

    return

def plot_boundary(means, covs, priors, ax=None, colour='black',
    xlimits=[-4,10], ylimits=[-4,10], num_points=100, line_width=2,
    show=True, save=False):
    '''
    Plots the decision boundary for Bayes classifier
    with Gaussian class conditionals

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    xx, yy = np.meshgrid(np.linspace(xlimits[0], xlimits[1], num_points),\
        np.linspace(ylimits[0], ylimits[1], num_points))
    data_grid = np.c_[xx.ravel(), yy.ravel()]

    zz = predict_gaussian_conditionals(data_grid, means, covs, priors).reshape(xx.shape)

    plt.contour(xx, yy, zz, [0.5], linewidths=line_width, colors=colour)
    if show:
        if save:
            plt.savefig(save + '.pdf', format='pdf')
        plt.show()

    return

# %% LOSS FUNCTIONS

def gmm_loglikelihood(data, priors, means, covs):
    '''
    Returns the log-likelihood of Gaussian mixture density
    evaluated on the training data

    :param data: input data
    :param priors: component priors
    :param means: component means
    :param covs: component covariances

    :return: log-likelihood of Gaussian mixture density

    '''

    num_components = len(priors)
    num_samples, dim = data.shape
    densities = np.zeros((num_samples, num_components))

    for k in range(num_components):
        for i in range(num_samples):
            densities[i, k] = multivariate_normal.pdf(data[i], \
                means[k, :], covs[k, :, :])

    return np.sum(np.log(densities.dot(priors)))

# %% ALGORITHMS

def score_gaussian_conditionals(x, mu, sigma, prior):
    '''
    Returns the score of the x belonging to class
    with mean mu and covariance sigma

    :param x: test feature
    :param mu: class mean
    :param sigma: class covariance
    :param prior: class prior

    :return: score for the class

    '''

    pres = np.linalg.inv(sigma)

    return -(np.log(prior) - (1.0/2.0)*np.log(np.linalg.det(pres)) - \
         (1.0/2.0)*(x-mu).T @ pres @ (x-mu)).flatten()[0]

def gmm_estep(data, prior, mean, cov):
    '''
    Runs E-step for GMMs
    Updates soft weights

    :param data: training samples
    :param prior: component weights
    :param mean: component means
    :param cov: component covariances

    :return: soft weights for M-step

    '''

    num_samples, dim = data.shape
    num_components = len(prior)

    gamma = np.zeros((num_samples, num_components))
    for i in range(num_samples):
        xi = data[i, :]
        den = 0
        for k in range(num_components):
            p = multivariate_normal.pdf(xi, mean[k,:], cov[k,:,:])
            den += prior[k] * p
        for k in range(num_components):
            gamma[i,k] = prior[k] * multivariate_normal.pdf(xi, mean[k,:], cov[k,:,:]) / den

    return gamma

def gmm_mstep(data, gamma):
    '''
    Runs M-step for GMMs
    Updates prior, means and covariances

    :param data: training samples
    :param gamma: weight matrix

    :return: updated priors, means, covariances

    '''

    num_samples, dim = data.shape
    num_components = np.size(gamma,1)

    means = np.zeros((num_components, dim))
    covs = np.zeros((num_components, dim, dim))
    priors = np.zeros(num_components)

    for k in range(num_components):
        den = np.sum(gamma, 0)[k]
        for i in range(num_samples):
            means[k] += gamma[i, k] * data[i]
        means[k] /= den
        for i in range(num_samples):
            left = np.reshape((data[i] - means[k]), (2,1))
            right = np.reshape((data[i] - means[k]), (1,2))
            covs[k] += gamma[i,k] * left * right
        covs[k] /= den
        priors[k] = den/num_samples
    
    return priors, means, covs

# %% TRAINING AND TESTING

def train_gaussian_conditionals(data):
    '''
    Trains 2 class classifier with Gaussian
    class conditionals

    :param data: training data
    :return: learnt mean and covariance of classes

    '''

    dim = data.shape[1]
    
    labels = data[:,dim-1]
    pos_samples = data[np.where(labels==1)][:,:dim-1]
    num_pos_samples = pos_samples.shape[0]

    neg_samples = data[np.where(labels==-1)][:,:dim-1]
    num_neg_samples = neg_samples.shape[0]

    # Compute sample mean
    pos_mean = np.mean(pos_samples, axis=0)
    neg_mean = np.mean(neg_samples, axis=0)

    # Compute sample covariance
    pos_cov = ((pos_samples-pos_mean).T).dot((pos_samples-pos_mean))/num_pos_samples
    neg_cov = ((neg_samples-neg_mean).T).dot((neg_samples-neg_mean))/num_neg_samples

    return pos_mean, neg_mean, pos_cov, neg_cov

def test_gaussian_conditionals(data, mean, cov, prior):
    '''
    Tests 2 class classifier with Gaussian
    class conditionals

    :param data: test data
    :param mean: class mean
    :param cov: class covariance
    :param prior: class prior

    :return: accuracy for test data

    '''

    dim = data.shape[1]
    true_labels = (data[:,dim-1]+1)/2       # Change labels to 0-1
    
    num_samples = data[:,:dim-1].shape[0]
    num_classes = len(mean)

    score = np.zeros((num_samples, num_classes))
    for i in tqdm(range(num_samples)):
        for classes in range(num_classes):
            score[i, classes] = (score_gaussian_conditionals(data[i,:dim-1],
                mean[classes], cov[classes], prior[classes]))

    predicted_labels = np.argmax(score, axis=1)

    confusion_mtx = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_mtx[i,j] = sum((true_labels==i) & (predicted_labels==j))

    return num_classes*confusion_mtx/num_samples

def predict_gaussian_conditionals(data, mean, cov, prior):
    '''
    Predicts labels for class classifier with
    Gaussian class conditionals =

    :param data: test data
    :param mean: class mean
    :param cov: class covariance
    :param prior: class prior

    :return: predicted labels

    '''

    num_samples = data.shape[0]
    num_classes = len(mean)

    score = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        for classes in range(num_classes):
            score[i, classes] = (score_gaussian_conditionals(data[i,:],
                mean[classes], cov[classes], prior[classes]))

    return np.argmax(score, axis=1)

def train_gmm(data, num_components, max_iter=100, tol=1e-3):
    '''
    Trains Gaussian Mixture Model using data

    :param data: training data
    :param num_components: number of Gaussian mixture components

    :return: learnt priors, means, covariances

    '''

    num_samples, dim = data.shape

    # Initialisation
    means = np.random.randn(num_components, dim-1)
    covs = np.zeros((num_components, dim-1, dim-1))
    for k in range(num_components):
        covs[k] = np.eye(dim-1, dim-1)
    priors = np.ones(num_components)/num_components

    cost = []
    cost_prev = -99999
    for iter in tqdm(range(max_iter)):

        gamma = gmm_estep(data[:,:dim-1], priors, means, covs)
        priors, means, covs = gmm_mstep(data[:,:dim-1], gamma)

        cost_iter = gmm_loglikelihood(data[:,:dim-1], priors, means, covs)
        cost.append(cost_iter)
        if (abs(cost_iter - cost_prev)**2 < tol):
            break
        cost_prev = cost_iter

    return priors, means, covs, cost