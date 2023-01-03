# importing package
import matplotlib.pyplot as plt
import numpy as np

# create data
x = np.arange(4)
y1 = [22, 16, 22, 15]
y2 = [22, 12, 28, 6]
y3 = [14, 10, 25, 20]
width = 0.2

fig = plt.figure(figsize=(8, 10))
axid = fig.add_subplot(2,1,1)


axhw = fig.add_subplot(2,1,2)


# plot data in grouped manner of bar type
axid.bar(x - 0.2, y1, width, color='blue')
axid.bar(x, y2, width, color='orange')
axid.bar(x + 0.2, y3, width, color='green')
for bars in axid.containers:
    axid.bar_label(bars)

axid.tick_params(axis='both', which='major', labelsize=12)
# axid.xlabel("Model-LM-DS combination")
axid.set_ylabel("#GE=1", fontsize=15)
# plt.setp(axid.get_xticklabels(), visible=False)
axid.legend(["orig", "ae_mlp_str_dcr", "ae_cnn"], loc='best', fontsize=11)
axid.set_title('ID leakage model', fontsize=20)


y1 = [44, 29, 38, 26]
y2 = [35, 25, 47, 4]
y3 = [37, 18, 35, 25]

# plot data in grouped manner of bar type
axhw.bar(x - 0.2, y1, width, color='blue')
axhw.bar(x, y2, width, color='orange')
axhw.bar(x + 0.2, y3, width, color='green')
for bars in axhw.containers:
    axhw.bar_label(bars)
axhw.set_ylabel("#GE=1", fontsize=15)
axhw.set_xlabel("Model-LM-DS combination", fontsize=15)
# axhw.set_xticklabels(x, ['MLP ID ASCADr', 'MLP ID DPAv42', 'CNN ID ASCADr', 'CNN ID DPAv42'])
plt.setp([axid, axhw], xticks=x, xticklabels=['MLP ASCADr', 'CNN ASCADr', 'MLP DPAv42', 'CNN DPAv42'])

axhw.set_title('HW leakage model', fontsize=20)
# plt.show()
axhw.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('tuning_both_orig_and_encod.png', format='png', dpi=300, bbox_inches='tight')