import os
import glob
import numpy as np


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def get_filename(folder_results, dataset_name, model_type, leakage_model, desync=False, gaussian_noise=False, time_warping=False):
    da_str = "_desync" if desync else ""
    da_str += "_gaussian_noise" if gaussian_noise else ""
    da_str += "_time_warping" if time_warping else ""

    file_count = 0
    for name in glob.glob(f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{da_str}_*.npz"):
        file_count += 1
    new_filename = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{da_str}_{file_count + 1}.npz"

    return new_filename


# utility function for SNR calculation
def CalculateSNR(l, IntermediateData, possibilities):
    trace_length = l.shape[1]
    mean = np.zeros([possibilities, trace_length])
    var = np.zeros([possibilities, trace_length])
    cpt = np.zeros(possibilities)
    i = 0

    for trace in l:
        # classify the traces based on its SBox output
        # then add the classified traces together
        mean[IntermediateData[i]] += trace
        var[IntermediateData[i]] += np.square(trace)
        # count the trace number for each SBox output
        cpt[IntermediateData[i]] += 1
        i += 1

    for i in range(possibilities):
        # average the traces for each SBox output
        mean[i] = mean[i] / cpt[i]
        # variance  = mean[x^2] - (mean[x])^2
        var[i] = var[i] / cpt[i] - np.square(mean[i])
    # Compute mean [var_cond] and var[mean_cond] for the conditional variances and means previously processed
    # calculate the trace variation in each points
    varMean = np.var(mean, 0)
    # evaluate if current point is stable for all S[p^k] cases
    MeanVar = np.mean(var, 0)
    return varMean / MeanVar

