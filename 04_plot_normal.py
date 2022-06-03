import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import core as c

from gaussian_processes_util import plot_gp

# mu = 0
# variance = 1
mu = 0.5
variance = 0.01
sigma = math.sqrt(variance)
# sigma = 2
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# x = np.linspace(0, 2, 100)
# c.plot_normaldistribution(x, mu, sigma, 'test')

# TODO pior
# from gaussian_processes_util import plot_gp

# # Finite number of points
# # X = np.arange(-5, 5, 0.2).reshape(-1, 1)
X = x.reshape(-1, 1)

# # Mean and covariance of the prior
mu = np.zeros(X.shape)
l = 0.1
# cov = c.kernel(X, X, l, sigma)

# # Draw three samples from the prior
# samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

# # Plot GP mean, uncertainty region and samples 
# plot_gp(mu, cov, X, samples=samples)
# plt.savefig('image_out/' + 'test' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

# TODO
# # Noise free training data
# # X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
# X_train = np.array([0.45, 0.6]).reshape(-1, 1)
# Y_train = np.sin(X_train)

# # Compute mean and covariance of the posterior distribution
# # mu_s, cov_s = c.posterior(X, X_train, Y_train)
# # mu_s, cov_s = c.posterior(X, X_train, Y_train, l, sigma, sigma_y=1e-8)
# mu_s, cov_s = c.posterior(X, X_train, Y_train, l, sigma, sigma_y=1e-5)

# samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
# plt.savefig('image_out/' + 'test' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

# TODO
# Noise free training data
# X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
X_train = np.array([0.45, 0.6]).reshape(-1, 1)
Y_train = np.sin(X_train)

# Compute mean and covariance of the posterior distribution
# mu_s, cov_s = c.posterior(X, X_train, Y_train)
# mu_s, cov_s = c.posterior(X, X_train, Y_train, l, sigma, sigma_y=1e-8)
l = 0.2
mu_s, cov_s = c.posterior(X, X_train, Y_train, l, sigma, sigma_y=1e-5)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
plt.savefig('image_out/' + 'test' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()