import matplotlib.pyplot as plt
import numpy as np
import sys
from src.neural_networks.models import *
from src.datasets.load_ascadr import *
from src.datasets.load_dpav42 import *
from src.datasets.paths import *
from tensorflow.keras.optimizers import *

model_type = ["cnn"] # ["mlp", "cnn"]
dataset_name = ["ascad-variable"]  #"ascad-variable", "ascadf", "dpa_v42"]
leakage_model = ["ID"] #["ID", 'HW']
home_folder = "./best_ascadf/TL"
# home_folder = "./hyperparams_1"
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
        latent_size = "_700"
    paramds = {}
    for ds in dataset_name:
        # print(f'dataset\tleakage model\tmodel type\tnb GE=1\tnb_runs\tmean traces\tmedian traces')
        paramlm = {}
        for lm in leakage_model:
            parammodel = {}
            for m_type in model_type:
                folder_results = f"{home_folder}/with_{ae_m}{latent_size}"
                # print(f'{ds}\t{lm}\t{m_type}', end='\t')
                for file_id in []:
                    filepath = f"{folder_results}/{ds}_{m_type}_{lm}_{file_id + 1}.npz"
                    npz_file = np.load(filepath, allow_pickle=True)
                    guessing_entropy = npz_file["GEs"]
                    min_GE_evol = np.min(guessing_entropy, axis=1)
                    min_GE = np.min(min_GE_evol)
                    epoch_i = np.argmin(min_GE_evol)
                    traces = np.where(guessing_entropy[epoch_i] == min_GE)[0][0]
                    print(f'{ds}\t{lm}\t{m_type}\t{npz_file["encoder_file"]}\t{npz_file["attack_model"]}\t{epoch_i+1}\t{min_GE}\t{traces}')
                for file_id in [12]:
                    filepath = f"{folder_results}/{ds}_{m_type}_{lm}_{file_id + 1}.npz"
                    npz_file = np.load(filepath, allow_pickle=True)
                    guessing_entropy = npz_file["GE"]
                    min_GE = np.min(guessing_entropy)
                    traces = np.argmin(guessing_entropy)
                    epochs = npz_file['dataset'].item()['epochs']
                    print(f'{ds}\t{lm}\t{m_type}\t{npz_file["encoder_file"]}\t{npz_file["attack_model"]}\t{epochs}\t{min_GE}\t{traces}')


