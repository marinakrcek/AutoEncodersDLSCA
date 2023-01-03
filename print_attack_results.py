import matplotlib.pyplot as plt
import numpy as np
import sys
from src.neural_networks.models import *
from src.datasets.load_ascadr import *
from src.datasets.load_dpav42 import *
from src.datasets.paths import *
from tensorflow.keras.optimizers import *


model_type = ["mlp"]  # ["mlp", "cnn"]
dataset_name = ["ascad-variable"] # ["ascadf", "dpa_v42", "ascad-variable"]
leakage_model = ["ID", 'HW']
nb_runs = 2
home_folder = "./best_encoded_dpav42"
# home_folder = "./hyperparams_1"
# home_folder = "/tudelft.net/staff-bulk/ewi/insy/CYS/mkrcek"
ae_model = ["ae_mlp_str_dcr"] # "orig", ["orig", "ae_mlp_str_dcr", "ae_cnn"]
# latent_size = "400" #"TC_CMA" #

optimizers = {"Adam": Adam, 'RMSprop': RMSprop, 'SGD': SGD, 'Adagrad': Adagrad}
params = {}
for ae_m in ae_model:
    print(ae_m)
    if ae_m == "orig":
        latent_size = ""
    else: latent_size = "_400"
    paramds = {}
    for ds in dataset_name:
        # print(f'dataset\tleakage model\tmodel type\tnb GE=1\tnb_runs\tmean traces\tmedian traces')
        paramlm = {}
        for lm in leakage_model:
            parammodel = {}
            for m_type in model_type:
                folder_results = f"{home_folder}/{m_type}_{ae_m}{latent_size}"
                GEs = []
                NTs = []
                trainable_params = []
                for file_id in range(0, nb_runs):
                    filepath = f"{folder_results}/{ds}_{m_type}_{lm}_{file_id+1}.npz"
                    npz_file = np.load(filepath, allow_pickle=True)
                    guessing_entropy = npz_file["GE"]
                    min_GE = np.min(guessing_entropy)
                    traces = np.argmin(guessing_entropy)
                    if min_GE > 1:
                        traces = -1
                    result = np.where(guessing_entropy <= 1)
                    GEs.append(min_GE)
                    NTs.append(traces)
                    # if min_GE > 1:
                    #     continue
                    # dataset_parameters = npz_file['dataset'].item()
                    # if ds == "dpa_v42":
                    #     class_name = ReadDPAV42
                    # elif ds == "ascad-variable":
                    #     class_name = ReadASCADr
                    #
                    # trace_folder = "./datasets"
                    # dataset = class_name(
                    #     dataset_parameters["n_profiling"],
                    #     dataset_parameters["n_attack"],
                    #     file_path=get_dataset_filepath(trace_folder, ds, dataset_parameters["npoi"]),
                    #     target_byte=dataset_parameters["target_byte"],
                    #     leakage_model=leakage_model,
                    #     first_sample=0,
                    #     number_of_samples=dataset_parameters["npoi"]
                    # )
                    # hp_values = npz_file['hp_values'].item()
                    # hp_values['optimizer'] = optimizers[hp_values['optimizer']]
                    # baseline_model = (dataset.classes, dataset.ns, hp_values) if model_type == "cnn" else mlp(dataset.classes,
                    #     dataset.ns,
                    #     hp_values)
                    # trainable_params.append(np.sum([np.prod(v.get_shape().as_list()) for v in baseline_model.trainable_variables]))
                    # # print(f'{ds}\t{lm}\t{m_type}\t{file_id+1}\t{min_GE}\ttrainable_variables\t{np.sum([np.prod(v.get_shape().as_list()) for v in baseline_model.trainable_variables])}', end="\t")
                    # # print(f'trainable_weights\t{np.sum([np.prod(v.get_shape().as_list()) for v in baseline_model.trainable_weights])}', end="\t")
                    # # print(f'non_trainable_variables\t {np.sum([np.prod(v.get_shape().as_list()) for v in baseline_model.non_trainable_variables])}', end="\t")
                    # # print(f'non_trainable_weights\t{np.sum([np.prod(v.get_shape().as_list()) for v in baseline_model.non_trainable_weights])}', end="\n")

                parammodel[m_type] = trainable_params
                # print(GEs)
                # print(NTs)
                GEs = np.array(GEs)
                NTs = np.array(NTs)
                nts = NTs[np.where(NTs > -1)]
                # print(nts)
                # print(np.mean(nts), np.median(nts))
                print(f'{ds}\t{lm}\t{m_type}\t{len(np.where(GEs <= 1)[0])}\t{len(GEs)}\t{np.mean(nts)}\t{np.median(nts)}')
            paramlm[lm] = parammodel
        paramds[ds] = paramlm
    params[ae_m] = paramds

# for ae_m in ae_model:
#     for ds in dataset_name:
#         per_dataset = []
#         for lm, lmdict in params[ae_m][ds].items():
#             for model, nbtv in lmdict.items():
#                 per_dataset += nbtv
#         print(f"mean for {ae_m} and {ds} is {np.mean(per_dataset)}")
# for ae_m in ae_model:
#     for lm in leakage_model:
#         per_lm = []
#         for ds, dsdict in params[ae_m].items():
#             for model, nbtv in dsdict[lm].items():
#                 per_lm += nbtv
#         print(f"mean for {ae_m} and {lm} is {np.mean(per_lm)}")
# for ae_m in ae_model:
#     for model in model_type:
#         per_model = []
#         for ds, dsdict in params[ae_m].items():
#             for lm, lmdict in dsdict.items():
#                 per_model += lmdict[model]
#         print(f"mean for {ae_m} and {model} is {np.mean(per_model)}")
#
#
