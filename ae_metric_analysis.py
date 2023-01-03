import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

model_types = ["ae_mlp", "ae_mlp_dcr", "ae_mlp_str_dcr", "ae_cnn"]
dataset_names = ["ascad-variable", "dpa_v42"]
leakage_model = "HW"
hiding = ""
nb_runs_iter = iter([160, 160, 160, 161, 140, 220, 160, 160])

figure = plt.gcf()

for model_type in model_types:
    folder_results = f"./{model_type}_TC_CMA"
    for dataset_name in dataset_names:

        mses = []
        files = []
        snr_orig_pred_HW = []
        snr_orig_pred_ID = []
        snr_orig_encod_HW = []
        snr_orig_encod_ID = []
        peak_diff_HW = []
        peak_diff_ID = []
        nb_runs = next(nb_runs_iter)
        for file_id in range(0, nb_runs):
            filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_{file_id + 1}.npz"
            npz_file = np.load(filepath, allow_pickle=True)
            mse_ds = npz_file["mse_ds"].item()
            mses.append(mse_ds)
            files.append(filepath)
            snr_HW = npz_file['snr_HW'].item()
            snr_ID = npz_file['snr_ID'].item()
            snr_orig_pred_HW.append(np.max(snr_HW['snr_orig']) - np.max(snr_HW['snr_pred']))
            snr_orig_pred_ID.append(np.max(snr_ID['snr_orig']) - np.max(snr_ID['snr_pred']))
            snr_orig_encod_HW.append(np.max(snr_HW['snr_orig']) - np.max(snr_HW['snr_encd']))
            snr_orig_encod_ID.append(np.max(snr_ID['snr_orig']) - np.max(snr_ID['snr_encd']))

            peak_loc_snr_orig_HW = np.argmax(snr_HW['snr_orig'])
            peak_diff_HW.append(snr_HW['snr_orig'][peak_loc_snr_orig_HW] - snr_HW['snr_pred'][peak_loc_snr_orig_HW])
            peak_loc_snr_orig_ID = np.argmax(snr_ID['snr_orig'])
            peak_diff_ID.append(snr_ID['snr_orig'][peak_loc_snr_orig_ID] - snr_ID['snr_pred'][peak_loc_snr_orig_ID])

            # plt.plot(snr_HW['snr_orig'])
            # plt.plot(snr_HW['snr_pred'])
            # plt.clf()
            # plt.plot(snr_ID['snr_orig'])
            # plt.plot(snr_ID['snr_pred'])
            # plt.clf()

        best = np.min(mses)
        best_i = np.argmin(mses)
        file = files[best_i]
        print("best AE", file)
        best_AE = np.load(file, allow_pickle=True)
        print(best_AE['hp_values'].item(), best_AE['mse_ds'].item())

        # plt.clf()
        figure.set_size_inches(5, 4)
        plt.scatter(snr_orig_pred_HW, mses, c='b', marker='.', label='orig-pred HW')
        plt.scatter(snr_orig_encod_HW, mses, c='b', marker='x', label='orig-encod HW')
        plt.scatter(snr_orig_pred_ID, mses, c='r', marker='.', label='orig-pred ID')
        plt.scatter(snr_orig_encod_ID, mses, c='r', marker='x', label='orig-encod ID')
        plt.title(f"max-max {model_type} {dataset_name}")
        plt.xlabel("SNR diff", fontsize=12)
        plt.ylabel("MSE", fontsize=12)
        plt.ylim(0, 2)
        plt.tight_layout()
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_mse_snr.png", dpi=300, bbox_inches='tight')
        # plt.close()

        plt.clf()
        plt.scatter(peak_diff_HW, mses, marker='.', color='b', label='HW')
        plt.scatter(peak_diff_ID, mses, marker='.', color='r', label='ID')
        plt.title(f"Orig peak diff {model_type} {dataset_name}")
        plt.xlabel("SNR diff", fontsize=12)
        plt.ylabel("MSE", fontsize=12)
        plt.ylim(0, 2)
        plt.tight_layout()
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_mse_peak_snr.png", dpi=300, bbox_inches='tight')
        plt.close()

        origi = np.argsort(mses)
        diffHW = np.argsort(peak_diff_HW)
        diffID = np.argsort(peak_diff_ID)
        print('sorting analysis with peak diff')
        print('HW\tID')
        HWs = []
        IDs = []
        for i in range(1, len(origi) + 1):
            cntHW = 0
            cntID = 0
            for j in range(0, i):
                if origi[j] == diffHW[j]:
                    cntHW += 1
                if origi[j] == diffID[j]:
                    cntID += 1
            print(f'{cntHW}/{i}\t{cntID}/{i}')
            HWs.append(cntHW)
            IDs.append(cntID)
        plt.clf()
        plt.plot(np.arange(1,11), HWs, marker='.', color='b', label='HW')
        plt.plot(np.arange(1,11), IDs, marker='.', color='r', label='ID')
        # plt.grid(True, which="both", ls="-")
        plt.title(f"#correct {model_type} {dataset_name}")
        plt.xlabel("Order ", fontsize=12)
        plt.ylabel("Correct", fontsize=12)
        # plt.xticks(range(1, len(ls_sizes) + 1), x_ticks)
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_order_correctness_orig_peak_diff.png", dpi=300, bbox_inches='tight')
        plt.close()

        diffHW = np.argsort(snr_orig_pred_HW)
        diffID = np.argsort(snr_orig_pred_ID)
        HWs = []
        IDs = []
        print('sorting analysis with max-max diff')
        print('HW\tID')
        for i in range(1, len(origi) + 1):
            cntHW = 0
            cntID = 0
            for j in range(0, i):
                if origi[j] == diffHW[j]:
                    cntHW += 1
                if origi[j] == diffID[j]:
                    cntID += 1
            print(f'{cntHW}/{i}\t{cntID}/{i}')
            HWs.append(cntHW)
            IDs.append(cntID)
        plt.clf()
        plt.plot(np.arange(1,11), HWs, marker='.', color='b', label='HW')
        plt.plot(np.arange(1,11), IDs, marker='.', color='r', label='ID')
        # plt.grid(True, which="both", ls="-")
        plt.title(f"#correct {model_type} {dataset_name}")
        plt.xlabel("Order", fontsize=12)
        plt.ylabel("Correct", fontsize=12)
        # plt.xticks(range(1, len(ls_sizes) + 1), x_ticks)
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_order_correctness_max_max_diff.png", dpi=300, bbox_inches='tight')
        plt.close()




        print(mses)
        print(snr_orig_pred_HW)
        print(snr_orig_pred_ID)
        print(snr_orig_encod_HW)
        print(snr_orig_encod_ID)
