import numpy as np
import matplotlib.pyplot as plt

data = np.load('trace_plot.npy')

# plt.plot(data)
# plt.show()

# draw plot
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

# plot mean and 2σ intervals
# plt.plot(X_new, mu, "r", lw=2, label="mean and 2σ region")
# plt.plot(X_new, mu + 2 * sd, "r", lw=1)
# plt.plot(X_new, mu - 2 * sd, "r", lw=1)
# plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)

# plot original data and true function
plt.plot(X, y, "ok", ms=3, alpha=1.0, label="observed data")
plt.plot(X, f_true, "dodgerblue", lw=3, label="true f")

plt.xlabel("x")
plt.ylim([-13, 13])
plt.title("predictive mean and 2σ interval")
plt.legend();