import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import core as c

# TODO import file
pdf = pd.read_csv('data/ChifromSentinal.csv')
pdf = pdf.iloc[:, 1].values
y = c.normalized_data(pdf, 0, 1)

num_iter = 10
min_η_per, max_η_per = 0.1, 2.0
# η_per = pm.HalfCauchy("η_per", beta=2, testval=1.0)
min_η_med, max_η_med = 0.05, 0.5
# η_med = pm.HalfCauchy("η_med", beta=0.5, testval=0.1)
min_η_noise, max_η_noise = 0.01, 0.2
# η_noise = pm.HalfNormal("η_noise", sd=0.5, testval=0.05)
min_σ, max_σ = 0.01, 0.3
# σ = pm.HalfNormal("σ",  sd=0.25, testval=0.05)
c.generate_parameters('yes', num_iter, min_η_per, max_η_per, min_η_med, max_η_med,\
						min_η_noise, max_η_noise, min_σ, max_σ, y)