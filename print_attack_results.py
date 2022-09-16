import matplotlib.pyplot as plt
import numpy as np
import sys

model_type = "mlp"
folder_results = f"./attack-{model_type}-ae-cnn"
dataset_name = "dpa_v42"
leakage_model = "HW"
hiding = ""

GEs = []
NTs = []
for file_id in range(0, 10):
    filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}_{file_id+1}.npz"
    npz_file = np.load(filepath, allow_pickle=True)
    guessing_entropy = npz_file["GE"]
    min_GE = np.min(guessing_entropy)
    traces = np.argmin(guessing_entropy)
    if min_GE > 1:
        traces = -1
    result = np.where(guessing_entropy <= 1)
    GEs.append(min_GE)
    NTs.append(traces)

print(GEs)
print(NTs)
GEs = np.array(GEs)
NTs = np.array(NTs)
print(len(np.where(GEs <= 1)[0]), len(GEs))
nts = NTs[np.where(NTs > -1)]
print(nts)
print(np.mean(nts), np.median(nts))
