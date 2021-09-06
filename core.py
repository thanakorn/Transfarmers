import numpy as np
import matplotlib.pyplot as plt
import random
# import pymc3 as pm
# import theano.tensor as tt
import matplotlib as mpl
import matplotlib.colors as colors

from skimage import exposure 
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from pymc3.gp.util import plot_gp_dist
# from sklearn.metrics import mean_squared_error

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

def main_GPR(y, η_per_i, η_med_i, η_noise_i, σ_i, count):
	# TODO training
	X = np.linspace(0, len(y), len(y))[:, None]
	with pm.Model() as model:
		# NOTE season
		# yearly periodic component x long term trend
		η_per = pm.HalfCauchy("η_per", beta=2, testval=η_per_i)
		ℓ_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075)
		period  = pm.Normal("period", mu=1, sd=0.05)
		ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=4, beta=3)
		cov_seasonal = η_per**2 * pm.gp.cov.Periodic(1, ℓ_psmooth, period) \
								* pm.gp.cov.Matern52(1, ℓ_pdecay)
		gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

		# NOTE intermidate
		# small/medium term irregularities
		η_med = pm.HalfCauchy("η_med", beta=0.5, testval=η_med_i)
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
		η_noise = pm.HalfNormal("η_noise", sd=0.5, testval=η_noise_i)
		ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
		σ = pm.HalfNormal("σ",  sd=0.25, testval=σ_i)
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
	plt.scatter(X_true, y, c='green', s=20, edgecolor='black', linewidths=1.5, alpha=0.9,\
				label="Observed data")
	plt.xlabel('Months')
	plt.ylabel('Pixels')
	# plt.ylim([-13, 13])
	plt.title('η_per: {:.4f}, η_med: {:.4f}, η_noise: {:.4f}, σ: {:.4f}'.format(η_per_i, η_med_i, η_noise_i, σ_i))
	plt.legend()
	save_file = 'trace' + str(count).zfill(5)
	plt.savefig('image_multiple_traces/' + save_file + '.png', format='png', bbox_inches='tight',\
				dpi=300, transparent=False, pad_inches=0.2)
	# plt.show()
	fig.clf()

	# TODO plot mean and +-SD
	mu, var = gp.predict(X_new, point=mp, diag=True)
	sd = np.sqrt(var)
	fig = plt.figure(figsize=(12, 5))
	ax = fig.gca()
	plt.plot(X_new, mu, "r", lw=2, label="mean and 2σ region")
	plt.plot(X_new, mu + 2 * sd, "r", lw=1)
	plt.plot(X_new, mu - 2 * sd, "r", lw=1)
	plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)
	plt.scatter(X_true, y, c='green', s=20, edgecolor='black', linewidths=1.5, alpha=0.9,\
				label="Observed data")
	plt.xlabel('Months')
	plt.ylabel('Pixels')
	# plt.ylim([-13, 13])
	plt.title('η_per: {:.4f}, η_med: {:.4f}, η_noise: {:.4f}, σ: {:.4f}'.format(η_per_i,\
				η_med_i, η_noise_i, σ_i))
	plt.legend()
	save_file = 'mean' + str(count).zfill(5)
	plt.savefig('image_means/' + save_file + '.png', format='png', bbox_inches='tight',\
				dpi=300, transparent=False, pad_inches=0.2)
	# plt.show()
	print('save image: ', save_file)
	fig.clf()

def generate_parameters(random_seed, num_iter, min_η_per, max_η_per, min_η_med, max_η_med,\
						min_η_noise, max_η_noise, min_σ, max_σ, data):
	η_per, η_med, η_noise, σ = create_random(random_seed, num_iter, min_η_per, max_η_per,\
									min_η_med, max_η_med, min_η_noise, max_η_noise, min_σ, max_σ)
	optimizer = {'η_per': list(η_per),
				'η_med': list(η_med),
				'η_noise': list(η_noise),
				'σ': list(σ)
				}
	num_loop = pow(num_iter, 4)
	count = 1 

	for η_per_i in optimizer['η_per']:
		for η_med_i in optimizer['η_med']:
			for η_noise_i in optimizer['η_noise']:
				for σ_i in optimizer['σ']:
					print('progress: {:.4f}'.format((count/num_loop) * 100))
					print(
						'η_per: ', η_per_i,
						'η_med: ', η_med_i,
						'η_noise: ', η_noise_i,
						'σ: ', σ_i
						)
					if count < 4402:
						print('already compute')
					elif count >= 4402:
						main_GPR(data, η_per_i, η_med_i, η_noise_i, σ_i, count)
					count += 50

def cleaning_data(data_tiff): #? cleaning data: remove nan and inf
	data_tiff = np.nan_to_num(data_tiff)
	# print('max value = ', data_tiff.max(), 'min value = ', data_tiff.min())
	return data_tiff, data_tiff.min(), data_tiff.max()

def clip(model, perc):
	(ROWs, COLs) = model.shape
	reshape2D_1D = model.reshape(ROWs*COLs)
	reshape2D_1D = np.sort(reshape2D_1D)
	if perc != 100:
		min_num = reshape2D_1D[ round(ROWs*COLs*(1-perc/100)) ]
		max_num = reshape2D_1D[ round((ROWs*COLs*perc)/100) ]
	elif perc == 100:
		min_num = min(model.flatten())
		max_num = max(model.flatten())

	if max_num < min_num:
		dummy = min_num
		min_num = max_num
		max_num = dummy
	else:
		pass
	return max_num, min_num, 

def plot_imshow(data, vmin, vmax, title_name, save_file):
	font_size = 12 # print
	plt.figure()
	ax = plt.gca()
	im = ax.imshow(data, cmap='Greys', vmin=vmin, vmax=vmax)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2%", pad=0.05)
	plt.colorbar(im, cax=cax, label='pixel value')
	ax.set_xlabel('pixel axis-x', fontdict={'fontsize': font_size})
	ax.set_ylabel('pixel axis-y', fontdict={'fontsize': font_size})
	ax.set_title(title_name)
	plt.tight_layout()
	plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def compute_kmeans(data, number_of_classes):
	vector_data = data.reshape(-1, 1) 
	random_centroid = 42 # interger number range 0-42
	kmeans = KMeans(n_clusters = number_of_classes, random_state = random_centroid).fit(vector_data)
	kmeans = kmeans.cluster_centers_[kmeans.labels_]
	kmeans = kmeans.reshape(data.shape)
	return kmeans

def plot_kmeans(data, title_name, save_file):
	font_size = 12 # print
	ax = plt.gca()
	# cmap = colors.ListedColormap(['#6475F2', '#aeaeb0', '#3D6C48', '#f3f59d'])
	# cmap = colors.ListedColormap(['#6475F2', '#3D6C48', '#aeaeb0'])
	cmap = colors.ListedColormap(['#aeaeb0', '#3D6C48', '#6475F2'])
	im = ax.imshow(data, cmap=cmap)
	# im = ax.imshow(data, cmap='rainbow')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2%", pad=0.05)
	plt.colorbar(im, cax=cax, label='classes')
	ax.set_xlabel('pixel axis-x', fontdict={'fontsize': font_size})
	ax.set_ylabel('pixel axis-y', fontdict={'fontsize': font_size})
	ax.set_title(title_name)
	plt.tight_layout()
	plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def image_segmentation(kmeans, min_pixel, max_pixel):
	# binary = np.where(kmeans >= kmeans.max()/2, 0, 1)
	# binary = convert_array_to_binary(kmeans)
	dummy = np.unique(kmeans)
	binary = np.where(kmeans > dummy[0], 0, 1)
	#? begin image segmentation
	label_out = label(binary, connectivity=1, return_num=False)
	for region in regionprops(label_out):
		(min_row, min_col, max_row, max_col) = region.bbox
		if region.area >= min_pixel and region.area <= max_pixel:
			binary[min_row:max_row, min_col:max_col] = 0
	################################# for QC plot
	# plt.figure(1)  
	# plt.imshow(binary)
	# plt.title('after cleaning band: ' + str(i))
	# plt.show()
	##################################
	# plt.tight_layout()
	# plt.savefig('image_out/'+save_file, format='svg', transparent=True)
	return binary
