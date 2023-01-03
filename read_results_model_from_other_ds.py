import matplotlib.pyplot as plt
import numpy as np
import sys
from src.neural_networks.models import *
from src.datasets.load_ascadr import *
from src.datasets.load_dpav42 import *
from src.datasets.paths import *
from tensorflow.keras.optimizers import *

model_type = ["mlp", "cnn"] # ["mlp", "cnn"]
dataset_name = ["ascadf",]  #"ascad-variable", "ascadf", "dpa_v42"]
leakage_model = ["ID", 'HW'] #["ID", 'HW']
nb_runs = 100
counter_start = 0
# home_folder = "./best_ascadf"
# home_folder = "./hyperparams_1"
home_folder = "./best_wo_std_ae_cnn_encoded_dpav42"
# home_folder = "/tudelft.net/staff-bulk/ewi/insy/CYS/mkrcek"
ae_model = ["ae_mlp_str_dcr"] #, "ae_mlp_str_dcr"]  #"orig", "ae_cnn"
# latent_size = "400" #"TC_CMA" #

optimizers = {"Adam": Adam, 'RMSprop': RMSprop, 'SGD': SGD, 'Adagrad': Adagrad}
params = {}
for ae_m in ae_model:
    print(ae_m)
    if ae_m == "orig":
        latent_size = ""
    else:
        latent_size = "_only_400"
    paramds = {}
    for ds in dataset_name:
        # print(f'dataset\tleakage model\tmodel type\tnb GE=1\tnb_runs\tmean traces\tmedian traces')
        paramlm = {}
        for lm in leakage_model:
            parammodel = {}
            for m_type in model_type:
                folder_results = f"{home_folder}/with_{ae_m}{latent_size}"
                GEs = []
                NTs = []
                other_ds_model = []
                trainable_params = []
                # print(f'{ds}\t{lm}\t{m_type}', end='\t')
                for file_id in range(counter_start, nb_runs):
                    filepath = f"{folder_results}/{ds}_{m_type}_{lm}_{file_id + 1}.npz"
                    npz_file = np.load(filepath, allow_pickle=True)
                    guessing_entropy = npz_file["GE"]
                    min_GE = np.min(guessing_entropy)
                    traces = np.argmin(guessing_entropy)
                    if min_GE > 1:
                        traces = -1
                    result = np.where(guessing_entropy <= 1)
                    GEs.append(min_GE)
                    NTs.append(traces)
                    other_ds_model.append(npz_file["attack_model"].item())

                GEs = np.array(GEs)
                NTs = np.array(NTs)
                nts = NTs[np.where(NTs > -1)]
                if not np.any(nts):
                    nts = NTs
                print(f'{ds}\t{lm}\t{m_type}\t{len(np.where(GEs <= 1)[0])}\t{len(GEs)}\t{np.mean(nts)}\t{np.median(nts)}', end='\t')
                GEs1 = np.where(GEs <= 1)[0]
                if not np.any(GEs1):
                    GEs1 = np.where(GEs <= np.min(GEs))[0]
                best_i = np.argmin(NTs[GEs1])
                print(np.unique(other_ds_model), end='\t')
                file_name = f"{folder_results}/{ds}_{m_type}_{lm}_{counter_start+GEs1[best_i] + 1}.npz"
                print(file_name, end='\t')
                print(GEs[GEs1[best_i]], end='\t')
                print(NTs[GEs1[best_i]], end='\t')
                npz_file = np.load(file_name, allow_pickle=True)
                guessing_entropy = npz_file["GE"]
                min_GE = np.min(guessing_entropy)
                traces = np.argmin(guessing_entropy)
                print(min_GE, end='\t')
                print(traces)


