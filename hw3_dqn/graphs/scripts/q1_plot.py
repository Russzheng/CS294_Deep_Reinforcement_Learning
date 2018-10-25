import csv
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import spline

mean_re = []
best_re = []
timestep = []

# load data
with open('data/plot.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      mean_re.append(float(row[0]))
      best_re.append(float(row[1]))
      timestep.append(float(row[2]))

fig = plt.figure()

# x, y limits
plt.axis([60000, timestep[-1]+10000, -22, 25])

# ticks
plt.xticks(np.arange(60000, timestep[-1]+100000, 20e4), np.arange(60000, timestep[-1]+10000, 20e4), rotation=45)
plt.yticks(np.arange(-22, 25, 2), np.arange(-22, 25, 2))

# smooth_step = np.linspace(timestep[0], timestep[-1], 30000)
# plt.plot(smooth_step, spline(timestep, mean_re, smooth_step), label='Mean Rewards')
# plt.plot(smooth_step, spline(timestep, best_re, smooth_step), label='Best Rewards')
plt.plot(timestep, mean_re, label='Mean Rewards')
plt.plot(timestep, best_re, label='Best Rewards')

# labels, title and legend
plt.xlabel('Timesteps')
plt.ylabel('Rewards Value')
plt.title('Pong Rewards vs Timesteps')
plt.grid()
plt.legend()
plt.show()