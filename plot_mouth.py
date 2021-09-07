import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import core as c

pdf = pd.read_csv('data/ChifromSentinal.csv')
pdf = pdf.iloc[:, 1].values
y = c.normalized_data(pdf, 0, 1)
y_true = c.normalized_data(pdf, 0, 1)
# y = y[:-12] # mj
# y_true = y_true[:-12]

x = np.linspace(0, 24, len(y))
plt.plot(x, y)
plt.savefig('image_out/' + 'all' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()