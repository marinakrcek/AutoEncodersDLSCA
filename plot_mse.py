import matplotlib.pyplot as plt
import numpy as np
import sys

model_type = "ae_mlp"
folder_results = f"./{model_type}"
dataset_name = "dpa_v42"
leakage_model = "HW"
hiding = ""

figure = plt.gcf()
figure.set_size_inches(5, 4)

for file_id in range(0,10):
    filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}_{file_id+1}.npz"
    npz_file = np.load(filepath, allow_pickle=True)
    # mse_ds = npz_file["mse_ds"].item()
    history = npz_file['history'].item()
    dataset_parameters = npz_file["dataset"].item()
    epochs = dataset_parameters['epochs']
    assert epochs, len(history['loss'])
    plt.plot(history['loss'], linewidth=1, label=f'{file_id+1}')
    plt.plot(history['val_loss'], linewidth=1, color=plt.gca().lines[-1].get_color(), linestyle='--')

plt.grid(True, which="both", ls="-")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.ylim(0,2)
plt.tight_layout()
plt.legend(loc='upper right', fontsize='small')
plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}.png",dpi=300, bbox_inches='tight')
plt.close()

