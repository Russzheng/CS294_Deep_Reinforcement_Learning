import csv
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import spline

mean_re_vanilla = []
mean_re_doubleqn = []
best_re_vanilla = []
best_re_doubleqn = []
timestep = []

# load data
# with open('data/plot.csv', 'r') as f:
with open('data/dqn_testing.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      mean_re_vanilla.append(float(row[0]))
      best_re_vanilla.append(float(row[1]))
      timestep.append(float(row[2]))

# with open('data/plot_ddqn.csv', 'r') as f:
with open('data/ddqn_testing.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      mean_re_doubleqn.append(float(row[0]))
      best_re_doubleqn.append(float(row[1]))

trim = min([len(mean_re_doubleqn), len(mean_re_doubleqn), len(timestep)])
mean_re_vanilla = mean_re_vanilla[:trim]
mean_re_doubleqn = mean_re_doubleqn[:trim]
best_re_vanilla = best_re_vanilla[:trim]
best_re_doubleqn = best_re_doubleqn[:trim]
timestep = timestep[:trim]
fig = plt.figure()

# x, y limits
# plt.axis([60000, timestep[-1]+10000, -22, 25])
plt.axis([60000, timestep[-1]+10000, -22, -17])

# ticks
plt.xticks(np.arange(60000, timestep[-1]+100000, 20e4), np.arange(60000, timestep[-1]+10000, 20e4), rotation=45)
# plt.yticks(np.arange(-22, 25, 2), np.arange(-22, 25, 2))
plt.yticks(np.arange(-22, -17, 0.2), np.arange(-22, -17, 0.2))

plt.plot(timestep, mean_re_vanilla, label='Vanilla DQN Mean')
plt.plot(timestep, mean_re_doubleqn, label='Double DQN Mean')
plt.plot(timestep, best_re_vanilla, label='Vanilla DQN Best')
plt.plot(timestep, best_re_doubleqn, label='Double DQN Best')

# labels, title and legend
plt.xlabel('Timesteps')
plt.ylabel('Rewards Value')
plt.title('Pong RAM Rewards Vanilla vs Double DQN')
plt.grid()
plt.legend()
plt.show()