import matplotlib.pyplot as plt
import numpy as np
import sys

model_type = "ae_mlp"
folder_results = f"./{model_type}"
dataset_name = "dpa_v42"
leakage_model = "HW"
hiding = ""

best_mse = sys.float_info.max
best_AE = None
best_file = None
for file_id in range(0, 10):
    filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}_{file_id+1}.npz"
    npz_file = np.load(filepath, allow_pickle=True)
    mse_ds = npz_file["mse_ds"].item()
    if mse_ds < best_mse:
        best_mse = mse_ds
        best_AE = npz_file
        best_file = filepath
    hp_values = npz_file['hp_values'].item()
    print(hp_values, mse_ds)
    # print(filepath, mse_ds)

print("best AE", best_file)
print(best_AE['hp_values'].item(), best_AE['mse_ds'].item())
# dataset_parameters = npz_file["dataset"].item()
# mse_ds = npz_file["mse_ds"].item()
# mse = npz_file['mse']
orig = best_AE['best_orig']
pred = best_AE['best_pred']

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
plt.plot(orig, linewidth=1, c='b', label='orig')
plt.plot(pred, linewidth=1, c='r', label='pred')
plt.grid(True, which="both", ls="-")
plt.xlabel("t", fontsize=12)
plt.ylabel("power", fontsize=12)
plt.tight_layout()
# plt.savefig(plot_filename)
# plt.close()
plt.legend()
plt.show()