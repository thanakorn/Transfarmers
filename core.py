import numpy as np
import matplotlib.pyplot as plt
import random
import pymc3 as pm
import theano.tensor as tt

from pymc3.gp.util import plot_gp_dist
from sklearn.metrics import mean_squared_error

def normalized_data(data, lowest_value, highest_value):
	data = (data - data.min()) / (data.max() - data.min())
	return data * (highest_value - lowest_value) + lowest_value

def plot_cross_section(data, lowest_value, highest_value, save_file):
	plt.figure(num=1, figsize=(12, 5))
	y = normalized_data(data, lowest_value, highest_value)
	x = np.linspace(0, len(y), len(y)) 
	plt.scatter(x=x, y=y, label='water surface (area)',  edgecolors='black', s=10, alpha=0.8)
	plt.legend(loc='upper left')
	plt.savefig('image_out/' + save_file, format='svg', transparent=True)
	plt.show()

def sub_1(num_iter, min_p, max_p):
	p = np.zeros(shape=num_iter, dtype=float)
	for i in range (num_iter):
		p[i] = random.random()
	return normalized_data(p, min_p, max_p)

def create_random(random_seed, num_iter, min_η_per, max_η_per, min_η_med, max_η_med,\
					min_η_noise, max_η_noise, min_σ, max_σ):
	if random_seed == 'yes':
		random.seed(10)
	elif random_seed == 'no':
		pass

	η_per = np.zeros(shape=num_iter, dtype=float)
	for i in range (num_iter):
		η_per[i] = random.random()
	η_per = normalized_data(η_per, min_η_per, max_η_per)

	η_med = np.zeros(shape=num_iter, dtype=float)
	for i in range (num_iter):
		η_med[i] = random.random()
	η_med = normalized_data(η_med, min_η_med, max_η_med)

	η_noise = np.zeros(shape=num_iter, dtype=float)
	for i in range (num_iter):
		η_noise[i] = random.random()
	η_noise = normalized_data(η_noise, min_η_noise, max_η_noise)

	σ = np.zeros(shape=num_iter, dtype=float)
	for i in range (num_iter):
		σ[i] = random.random()
	σ = normalized_data(σ, min_σ, max_σ)

	return η_per, η_med, η_noise, σ   

def generate_parameters(random_seed, num_iter, min_η_per, max_η_per, min_η_med, max_η_med,\
						min_η_noise, max_η_noise, min_σ, max_σ):
    # num_loop = num_lr*num_wd*num_mo	
	η_per, η_med, η_noise, σ = create_random(random_seed, num_iter, min_η_per, max_η_per,\
									min_η_med, max_η_med, min_η_noise, max_η_noise, min_σ, max_σ)
	optimizer = {'η_per': list(η_per),
				'η_med': list(η_med),
				'η_noise': list(η_noise),
				'σ': list(σ)
				}

	for η_per_i in optimizer['η_per']:
		for η_med_i in optimizer['η_med']:
			for η_noise_i in optimizer['η_noise']:
				for σ_i in optimizer['σ']:
					print('η_per: ', η_per_i,
						'η_med: ', η_med_i,
						'η_noise: ', η_noise_i,
						'σ: ', σ_i)

def main_GPR(y, p1, p2):
	# TODO training
	X = np.linspace(0, len(y), len(y))[:, None]
	with pm.Model() as model:
		# NOTE season
		# yearly periodic component x long term trend
		η_per = pm.HalfCauchy("η_per", beta=2, testval=1.0)
		ℓ_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075)
		period  = pm.Normal("period", mu=1, sd=0.05)
		ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=4, beta=3)
		cov_seasonal = η_per**2 * pm.gp.cov.Periodic(1, ℓ_psmooth, period) \
								* pm.gp.cov.Matern52(1, ℓ_pdecay)
		gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

		# NOTE intermidate
		# small/medium term irregularities
		η_med = pm.HalfCauchy("η_med", beta=0.5, testval=0.1)
		ℓ_med = pm.Gamma("ℓ_med", alpha=2, beta=0.75)
		α = pm.Gamma("α", alpha=5, beta=2) 
		cov_medium = η_med**2 * pm.gp.cov.RatQuad(1, ℓ_med, α)
		gp_medium = pm.gp.Marginal(cov_func=cov_medium)

		# NOTE long term
		# long term trend
		η_trend = pm.HalfCauchy("η_trend", beta=1, testval=2.0)
		ℓ_trend = pm.Gamma("ℓ_trend", alpha=4, beta=0.1)
		cov_trend = η_trend**2 * pm.gp.cov.ExpQuad(1, ℓ_trend)

		# positive trend
		gp_trend = pm.gp.Marginal(cov_func=cov_trend)   

		# NOTE noise
		# noise model
		η_noise = pm.HalfNormal("η_noise", sd=0.5, testval=0.05)
		ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
		σ = pm.HalfNormal("σ",  sd=0.25, testval=0.05)
		cov_noise = η_noise**2 * pm.gp.cov.Matern52(1, ℓ_noise) +\
					pm.gp.cov.WhiteNoise(σ)

		gp = gp_seasonal + gp_medium + gp_trend

		y_ = gp.marginal_likelihood("y", X=X, y=y, noise=cov_noise)
		mp = pm.find_MAP()

	# TODO predicting
	X_new = np.linspace(0, len(y)+20, len(y)+600)[:, None]
	# add the GP conditional to the model, given the new X values
	with model:
		f_pred = gp.conditional("f_pred", X_new)
	# To use the MAP values, you can just replace the trace with a length-1 list with `mp`
	with model:
		pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=50)

	X_true = np.linspace(0, len(y), len(y))[:, None]
	# TODO plot multiple traces
	fig = plt.figure(figsize=(12, 5))
	ax = fig.gca()
	plot_gp_dist(ax, pred_samples["f_pred"], X_new)
	plt.plot(X_true, y, "ok", ms=3, alpha=0.5, label="Observed data")
	plt.xlabel('Months')
	plt.ylabel('Pixels')
	# plt.ylim([-13, 13])
	plt.title('Time Series')
	plt.legend()
	plt.savefig('image_out/' + 'time_series_mj.svg', format='svg', transparent=True)
	plt.show()

	# TODO plot mean and +-SD
	mu, var = gp.predict(X_new, point=mp, diag=True)
	sd = np.sqrt(var)
	fig = plt.figure(figsize=(12, 5))
	ax = fig.gca()
	plt.plot(X_new, mu, "r", lw=2, label="mean and 2σ region")
	plt.plot(X_new, mu + 2 * sd, "r", lw=1)
	plt.plot(X_new, mu - 2 * sd, "r", lw=1)
	plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)
	plt.plot(X_true, y, "ok", ms=3, alpha=0.5, label="Observed data")
	plt.xlabel('Months')
	plt.ylabel('Pixels')
	# plt.ylim([-13, 13])
	plt.title('Mean and SD')
	plt.legend()
	plt.savefig('image_out/' + 'credible_mj.svg', format='svg', transparent=True)
	plt.show()