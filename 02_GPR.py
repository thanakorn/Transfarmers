import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import core as c

# set the seed
np.random.seed(1)

# TODO import file
# pdf = pd.read_csv('data/new_area_normalized.csv')
pdf = pd.read_csv('data/ChifromSentinal.csv')
pdf = pdf.iloc[:, 1].values
# y = c.normalized_data(pdf, -1, 1)
# y_true = c.normalized_data(pdf, -1, 1)
y = c.normalized_data(pdf, 0, 1)
y_true = c.normalized_data(pdf, 0, 1)
# y = y[:-6] # anan
y = y[:-4] # mj

# TODO training
X = np.linspace(0, len(y), len(y))[:, None]
with pm.Model() as model:
    # yearly periodic component x long term trend
    η_per = pm.HalfCauchy("η_per", beta=2, testval=1.0)
    ℓ_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075)
    period  = pm.Normal("period", mu=1, sd=0.05)
    ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=4, beta=3)
    cov_seasonal = η_per**2 * pm.gp.cov.Periodic(1, ℓ_psmooth, period) \
                            * pm.gp.cov.Matern52(1, ℓ_pdecay)
    gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

    # small/medium term irregularities
    η_med = pm.HalfCauchy("η_med", beta=0.5, testval=0.1)
    ℓ_med = pm.Gamma("ℓ_med", alpha=2, beta=0.75)
    α = pm.Gamma("α", alpha=5, beta=2) 
    cov_medium = η_med**2 * pm.gp.cov.RatQuad(1, ℓ_med, α)
    gp_medium = pm.gp.Marginal(cov_func=cov_medium)

    # long term trend
    η_trend = pm.HalfCauchy("η_trend", beta=1, testval=2.0)
    ℓ_trend = pm.Gamma("ℓ_trend", alpha=4, beta=0.1)
    cov_trend = η_trend**2 * pm.gp.cov.ExpQuad(1, ℓ_trend)

    # positive trend
    gp_trend = pm.gp.Marginal(cov_func=cov_trend)   

    # noise model
    η_noise = pm.HalfNormal("η_noise", sd=0.5, testval=0.05)
    ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
    σ = pm.HalfNormal("σ",  sd=0.25, testval=0.05)
    cov_noise = η_noise**2 * pm.gp.cov.Matern52(1, ℓ_noise) +\
                pm.gp.cov.WhiteNoise(σ)

    # gp = gp_seasonal + gp_medium + gp_trend
    # gp = gp_seasonal + gp_trend
    gp = gp_seasonal

    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=cov_noise)
    mp = pm.find_MAP()

# TODO predicting
X_new = np.linspace(0, len(y)+20, len(y)+600)[:, None]
# add the GP conditional to the model, given the new X values
with model:
    f_pred = gp.conditional("f_pred", X_new)
# To use the MAP values, you can just replace the trace with a length-1 list with `mp`
with model:
    # pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=2000)
    # pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=1000)
    # pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=500)
    # pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=500)
    pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=50)

filepath_p = 'test'
np.save(filepath_p, pred_samples["f_pred"][:, :])

X_true = np.linspace(0, len(y_true), len(y_true))[:, None]
# TODO plot multiple traces
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
# plot the samples from the gp posterior with samples and shading
from pymc3.gp.util import plot_gp_dist
plot_gp_dist(ax, pred_samples["f_pred"], X_new)
# plot the data and the true latent function
# plt.plot(X, y, "ok", ms=3, alpha=0.5, label="Observed data")
plt.plot(X_true, y_true, "ok", ms=3, alpha=0.5, label="Observed data")
plt.xlabel('Months')
# plt.ylim([-13, 13])
plt.title('Time Series')
plt.legend()
plt.savefig('image_out/' + 'time_series_mj' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

# TODO plot mean and +-SD
# predict
mu, var = gp.predict(X_new, point=mp, diag=True)
sd = np.sqrt(var)

# draw plot
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
# plot mean and 2σ intervals
plt.plot(X_new, mu, "r", lw=2, label="mean and 2σ region")
plt.plot(X_new, mu + 2 * sd, "r", lw=1)
plt.plot(X_new, mu - 2 * sd, "r", lw=1)
plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)
# plot original data and true function
plt.plot(X_true, y_true, "ok", ms=3, alpha=0.5, label="Observed data")
# plt.plot(X, y, "ok", ms=3, alpha=1.0, label="observed data")
plt.xlabel('Months')
# plt.ylim([-13, 13])
plt.title('Mean and SD')
plt.legend()
plt.savefig('image_out/' + 'SD' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()