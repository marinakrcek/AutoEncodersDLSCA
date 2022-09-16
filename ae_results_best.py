import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

model_type = "ae_cnn"
folder_results = f"./{model_type}_100_runs"
dataset_name = "dpa_v42"
leakage_model = "HW"
hiding = ""

mses = []
files = []

info = defaultdict(list)
for file_id in range(0, 9):
    filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}_{file_id+1}.npz"
    npz_file = np.load(filepath, allow_pickle=True)
    mse_ds = npz_file["mse_ds"].item()
    mses.append(mse_ds)
    files.append(filepath)
    hp_values = npz_file['hp_values'].item()
    print(hp_values, mse_ds)
    # print(filepath, mse_ds)
    info[hp_values['latent_dim']].append([filepath, mse_ds])


best = np.min(mses)
best_i = np.argmin(mses)
file = files[best_i]
print("best AE", file)
best_AE = np.load(file, allow_pickle=True)
print(best_AE['hp_values'].item(), best_AE['mse_ds'].item())

sorted_is = np.argsort(mses)
files = np.array(files)
mses = np.array(mses)
print('best 10')
for f, e in zip(files[sorted_is[:10]], mses[sorted_is[:10]]):
    print(f)
    ff = np.load(f, allow_pickle=True)
    print(ff['hp_values'], e)
# orig = best_AE['best_orig']
# pred = best_AE['best_pred']

# history = npz_file['history'].item()
# hp_values = npz_file['hp_values'].item()
# epochs = dataset_parameters['epochs']
# assert epochs, len(history['loss'])

# figure = plt.gcf()
# figure.set_size_inches(5, 4)
# plt.plot(history['loss'], linewidth=1, c='b', label='loss')
# plt.plot(history['val_loss'], linewidth=1, c='r', label='val_loss')
# plt.grid(True, which="both", ls="-")
# plt.xlabel("Epochs", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.tight_layout()
# # plt.savefig(plot_filename)
# plt.show()
# plt.close()

figure = plt.gcf()
figure.set_size_inches(5, 4)
# plt.plot(orig, linewidth=1, c='b', label='orig')
# plt.plot(pred, linewidth=1, c='r', label='pred')
# plt.grid(True, which="both", ls="-")
# plt.xlabel("t", fontsize=12)
# plt.ylabel("power", fontsize=12)
# plt.tight_layout()
# # plt.savefig(plot_filename)
# # plt.close()
# plt.legend()
# plt.show()

plt.clf()
ls_sizes = sorted(list(info.keys()))
for_plot = []
x_ticks = []
for ls_s in ls_sizes:
    mses = [ii[1] for ii in info[ls_s]]
    for_plot.append(mses)
    x_ticks.append(f"{ls_s} ({len(mses)})")
plt.boxplot(for_plot)
plt.grid(True, which="both", ls="-")
plt.xlabel("Latent space sizes", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.xticks(range(1, len(ls_sizes)+1), x_ticks)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_mse_per_z_size.png",dpi=300, bbox_inches='tight')
plt.close()