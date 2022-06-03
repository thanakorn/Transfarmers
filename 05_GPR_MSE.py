import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import core as c

# TODO import file
pdf = pd.read_csv('data/ChifromSentinal.csv')
pdf = pdf.iloc[:, 1].values
y = c.normalized_data(pdf, 0, 1)
y_true = c.normalized_data(pdf, 0, 1)
test_data = y

# TODO test data
y = y[:-12] # mj
y_true = y_true[:-12]
# y_true[:-12] = -1

# TODO training
X = np.linspace(0, len(y), len(y))[:, None]
# with pm.Model() as model:
#     η_per = pm.HalfCauchy("η_per", beta=2, testval=1.0)
#     ℓ_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075)
#     period  = pm.Normal("period", mu=4, sd=0.05)
#     ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=10, beta=1)
#     cov_seasonal = η_per**2 * pm.gp.cov.Periodic(1, ℓ_psmooth, period) * pm.gp.cov.Matern52(1, ℓ_pdecay)
#     gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)
#     η_noise = pm.HalfNormal("η_noise", sd=0.5, testval=0.05)
#     ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
#     σ = pm.HalfNormal("σ",  sd=0.25, testval=0.05)
#     cov_noise = η_noise**2 * pm.gp.cov.Matern52(1, ℓ_noise) + pm.gp.cov.WhiteNoise(σ)
#     gp = gp_seasonal
#     y_ = gp.marginal_likelihood("y", X=X, y=y, noise=cov_noise)
#     mp = pm.find_MAP()

with pm.Model() as model:
    n = pm.HalfCauchy("n", beta=2, testval=1.0)
    l = pm.Gamma("l", alpha=10, beta=0.075)
    p  = pm.Normal("p", mu=4, sd=0.05)
    ll = pm.Gamma("ll ", alpha=10, beta=1)
    cov_1 = n**2 * pm.gp.cov.Periodic(1, ll, p) * pm.gp.cov.Matern52(1, l)
    gp_seasonal = pm.gp.Marginal(cov_func=cov_1)
    nn = pm.HalfNormal("nn", sd=0.5, testval=0.05)
    nnn = pm.Gamma("nnn", alpha=2, beta=4)
    s = pm.HalfNormal("s",  sd=0.25, testval=0.05)
    cov_n = nn**2 * pm.gp.cov.Matern52(1, nnn) + pm.gp.cov.WhiteNoise(s)
    gp = gp_seasonal
    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=cov_n)
    mp = pm.find_MAP()

# TODO predicting
X_new = np.linspace(0, len(y)+20, len(y)+600)[:, None]
# filepath_p = 'test'
# np.save(filepath_p, mp)
X_true = np.linspace(0, len(y_true), len(y_true))[:, None]

# TODO plot mean and +-SD
mu, var = gp.predict(X_new, point=mp, diag=True)
sd = np.sqrt(var)
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
plt.plot(X_new, mu, "r", lw=2, label="mean")
plt.plot(X_new, mu + 2 * sd, "r", lw=1)
plt.plot(X_new, mu - 2 * sd, "r", lw=1)
plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)
plt.plot(X_true, y_true, "ok", ms=3, alpha=0.5, label="Observed data")
# plt.plot(X_true, y_true, "ob", ms=5, alpha=0.5, label="test data")
# plt.plot(X_true, test_data, "ok", ms=3, alpha=0.5, label="Observed data")
# plt.ylim([-0.5, 1])
plt.xlabel('Months')
plt.title('Gaussian Process Regression')
plt.legend()
plt.savefig('image_out/' + 'test' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

# TODO regression
# from scipy import signal

# mu, var = gp.predict(X_new, point=mp, diag=True)
# # mu = signal.resample(mu, len(test_data))
# test_data = signal.resample(test_data, len(mu))
# x1 = np.linspace(0, len(X_true), len(mu))
# # x2 = np.linspace(0, len(X_new), len(X_new))
# plt.plot(x1, mu, linewidth=5, color='#EC7063')
# plt.plot(x1, test_data, 'og', ms=10, alpha=0.8)
# # plt.ylim([-])
# plt.xlabel('Months')
# plt.title('Root Mean Square of Test Datapoints')
# # plt.plot(mu, 'ob', ms=10, alpha=0.8)
# plt.show()

# from sklearn.metrics import mean_squared_error

# mu, var = gp.predict(X_new, point=mp, diag=True)
# test_data = signal.resample(test_data, len(mu))
# print(mean_squared_error(test_data[881:], mu[881:]))
# plt.plot(mu, linewidth=5, color='#EC7063')
# plt.plot(test_data, 'og', ms=10, alpha=0.8)
# plt.show()