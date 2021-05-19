import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import core as c

# TODO import file
pdf = pd.read_csv('data/ChifromSentinal.csv')
pdf = pdf.iloc[:, 1].values
y = c.normalized_data(pdf, 0, 1)

c.generate_parameters()