import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

methods = ["RK2", "RK4", "RK6", "AB4", "BDF2"]

# trial 1 data
single1 = [0.001640, 0.003427, 0.005433, 0.002821, 0.004964]
system1 = [0.092300, 0.152817, 0.272787, 0.206642, 0.354923]

# trial 2 data
single2 = [0.001664, 0.003574, 0.008804, 0.002993, 0.005362]
system2 = [0.084223, 0.202134, 0.275138, 0.212298, 0.368964]

# formatting
T1_COLOR = "tab:blue"
T2_COLOR = "tab:orange"
BAR_WIDTH = 0.4
FONT_SIZE = 16

positions = []
values1 = []
values2 = []
labels = []

x = 0

# single odes
for i, method in enumerate(methods):
    positions.append(x)
    values1.append(single1[i])
    values2.append(single2[i])
    labels.append(f"{method} Single")
    x += 1

x += 1

# system odes
for i, method in enumerate(methods):
    positions.append(x)
    values1.append(system1[i])
    values2.append(system2[i])
    labels.append(f"{method} System")
    x += 1


positions = np.array(positions)

# plotting!!!
plt.figure()

plt.bar(positions - BAR_WIDTH/2, values1, width=BAR_WIDTH, color=T1_COLOR, label="Trial 1")
plt.bar(positions + BAR_WIDTH/2, values2, width=BAR_WIDTH, color=T2_COLOR, label="Trial 2")

# formatting
plt.xticks(positions, labels, rotation=90, fontsize=FONT_SIZE)
plt.ylabel("Runtime (seconds)", fontsize=FONT_SIZE)
# plt.title("Runtime Comparison Across Methods (Trial 1 vs Trial 2)", fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)

plt.legend(fontsize=FONT_SIZE)

plt.grid(axis="y")

plt.tight_layout()
plt.show()