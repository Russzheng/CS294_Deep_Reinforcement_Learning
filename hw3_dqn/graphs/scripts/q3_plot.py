import csv
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import spline

mean_re_vanilla_lr1e4 = []
mean_re_vanilla_lr1e3 = []
mean_re_vanilla_lr1e2 = []
mean_re_vanilla_lr1e5 = []
timestep = []

# load data
with open('data/ddqn_testing.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      mean_re_vanilla_lr1e4.append(float(row[0]))
      timestep.append(float(row[2]))

with open('data/plot_lr1e-5.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      mean_re_vanilla_lr1e5.append(float(row[0]))

with open('data/plot_lr1e-3.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      mean_re_vanilla_lr1e3.append(float(row[0]))

with open('data/plot_lr1e-2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      mean_re_vanilla_lr1e2.append(float(row[0]))
      
trim = min([len(mean_re_vanilla_lr1e4), len(mean_re_vanilla_lr1e3), len(mean_re_vanilla_lr1e2), len(mean_re_vanilla_lr1e5)])
mean_re_vanilla_lr1e4 = mean_re_vanilla_lr1e4[:trim]
mean_re_vanilla_lr1e3 = mean_re_vanilla_lr1e3[:trim]
mean_re_vanilla_lr1e2 = mean_re_vanilla_lr1e2[:trim]
mean_re_vanilla_lr1e5 = mean_re_vanilla_lr1e5[:trim]
timestep = timestep[:trim]
fig = plt.figure()

# x, y limits
# plt.axis([60000, timestep[-1]+10000, -22, 25])
plt.axis([60000, timestep[-1]+10000, -22, -16])

# ticks
plt.xticks(np.arange(60000, timestep[-1]+100000, 20e4), np.arange(60000, timestep[-1]+10000, 20e4), rotation=45)
plt.yticks(np.arange(-22, -16, 0.5), np.arange(-22, -16, 0.5))

plt.plot(timestep, mean_re_vanilla_lr1e4, label='lr1e-4')
plt.plot(timestep, mean_re_vanilla_lr1e3, label='lr1e-3')
plt.plot(timestep, mean_re_vanilla_lr1e2, label='lr1e-2')
plt.plot(timestep, mean_re_vanilla_lr1e5, label='lr1e-5')

# labels, title and legend
plt.xlabel('Timesteps')
plt.ylabel('Rewards Value')
plt.title('Pong Ram Rewards Learning Rate Comparison (Double DQN)')
plt.grid()
plt.legend()
plt.show()
