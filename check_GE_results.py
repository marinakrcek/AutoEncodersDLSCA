import matplotlib.pyplot as plt
import numpy as np
import sys

model_type = "cnn"
folder_results = f"./{model_type}_orig_100_runs"
dataset_name = "ascad-variable"
leakage_model = "ID"
hiding = ""
nb_runs = 100
best_mse = sys.float_info.max
best_AE = None
best_file = None
figure = plt.gcf()
figure.set_size_inches(5, 4)
for file_id in range(0, nb_runs):
    filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}_{file_id+1}.npz"
    npz_file = np.load(filepath, allow_pickle=True)
    guessing_entropy = npz_file["GE"]
    # if mse_ds < best_mse:
    #     best_mse = mse_ds
    #     best_AE = npz_file
    #     best_file = filepath
    # hp_values = npz_file['hp_values'].item()
    # print(hp_values, mse_ds)

    plt.plot(guessing_entropy, linewidth=1)
plt.grid(True, which="both", ls="-")
plt.xlabel("Traces", fontsize=12)
plt.ylabel("Guessing Entropy", fontsize=12)
plt.ylim(0, 256)
plt.tight_layout()
plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}.png",dpi=300, bbox_inches='tight')
plt.close()
# plt.show()
