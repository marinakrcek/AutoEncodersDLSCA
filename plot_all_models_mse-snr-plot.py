import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict
import matplotlib.cm as cm
from scipy import stats

# model_types = ["ae_mlp", "ae_mlp_dcr", "ae_cnn"]
# dataset_names = ["ascad-variable", "dpa_v42"]
# leakage_model = "HW"
# hiding = ""
# nb_runs = 20
model_types = ["ae_mlp", "ae_mlp_dcr", "ae_mlp_str_dcr", "ae_cnn"]
dataset_names = ["dpa_v42", "ascad-variable"]
leakage_model = "HW"
hiding = ""
nb_runs_iter = iter([160, 161, 220, 160, 160, 160, 140, 160])
max_runs = 20


for dataset_name in dataset_names:
    figure = plt.gcf()
    plt.grid(True, which="both", ls="-")
    colors = iter(['b', 'r', 'y', 'g'])
    peak_snr_diff_all_HW = []
    peak_snr_diff_all_ID = []
    mse_all = []
    for model_type in model_types:
        folder_results = f"./{model_type}_TC_CMA"

        mses = []
        files = []
        snr_orig_pred_HW = []
        snr_orig_pred_ID = []
        snr_orig_encod_HW = []
        snr_orig_encod_ID = []
        peak_diff_HW = []
        peak_diff_ID = []

        latent20s = 0
        nb_runs = next(nb_runs_iter)
        for file_id in range(0, nb_runs):
            filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_{file_id + 1}.npz"
            npz_file = np.load(filepath, allow_pickle=True)
            hp_values = npz_file['hp_values'].item()
            if hp_values['latent_dim'] == 20:
                latent20s += 1
            if hp_values['latent_dim'] == 20 and latent20s > max_runs:
                continue
            mse_ds = npz_file["mse_ds"].item()
            mses.append(mse_ds)
            files.append(filepath)
            snr_HW = npz_file['snr_HW'].item()
            snr_ID = npz_file['snr_ID'].item()
            snr_orig_pred_HW.append(np.max(snr_HW['snr_orig']) - np.max(snr_HW['snr_pred']))
            snr_orig_pred_ID.append(np.max(snr_ID['snr_orig']) - np.max(snr_ID['snr_pred']))
            # snr_orig_encod_HW.append(np.max(snr_HW['snr_orig']) - np.max(snr_HW['snr_encd']))
            # snr_orig_encod_ID.append(np.max(snr_ID['snr_orig']) - np.max(snr_ID['snr_encd']))

            peak_loc_snr_orig_HW = np.argmax(snr_HW['snr_orig'])
            peak_diff_HW.append(snr_HW['snr_orig'][peak_loc_snr_orig_HW] - snr_HW['snr_pred'][peak_loc_snr_orig_HW])
            peak_loc_snr_orig_ID = np.argmax(snr_ID['snr_orig'])
            peak_diff_ID.append(snr_ID['snr_orig'][peak_loc_snr_orig_ID] - snr_ID['snr_pred'][peak_loc_snr_orig_ID])

        best = np.min(mses)
        best_i = np.argmin(mses)
        file = files[best_i]
        # print("best AE", file)
        best_AE = np.load(file, allow_pickle=True)
        # print(best_AE['hp_values'].item(), best_AE['mse_ds'].item())

        peak_snr_diff_all_HW.append(peak_diff_HW)
        peak_snr_diff_all_ID.append(peak_diff_ID)
        mse_all.append(mses)

        mses = np.array(mses)
        mse_threshold = np.where(mses > 0.25)[0]
        mset = np.where(mses < 0.375)[0]
        mse_threshold = np.intersect1d(mse_threshold, mset)
        peak_diff_HW = np.array(peak_diff_HW)
        peak_diff_ID = np.array(peak_diff_ID)
        pcorr = stats.pearsonr(peak_diff_HW[mse_threshold], mses[mse_threshold])
        print(f'{dataset_name}\t{model_type}\tHW\t{pcorr.statistic}\t{pcorr.pvalue}')
        pcorr = stats.pearsonr(peak_diff_ID[mse_threshold], mses[mse_threshold])
        print(f'{dataset_name}\t{model_type}\tID\t{pcorr.statistic}\t{pcorr.pvalue}')

        figure.set_size_inches(5, 4)

    #     c = next(colors)
    #     plt.scatter(snr_orig_encod_HW, mses, c=c, marker='.', label=f'{model_type} HW')
    #     plt.scatter(snr_orig_encod_ID, mses, c=c, marker='x', label=f'{model_type} ID')
    # plt.title(f"max_orig-max_encod {dataset_name}")
    # plt.xlabel("SNR diff", fontsize=12)
    # plt.ylabel("MSE", fontsize=12)
    # plt.ylim(0, 2)
    # plt.tight_layout()
    # plt.legend(loc='upper right', fontsize='small')
    # plt.savefig(f"./paper_figures/{dataset_name}_mse_orig_encod_diff_snr.png", dpi=300, bbox_inches='tight')
    # plt.close()

    #     c=next(colors)
    #     plt.scatter(snr_orig_pred_HW, mses, c=c, marker='.', label=f'{model_type} HW')
    #     plt.scatter(snr_orig_pred_ID, mses, c=c, marker='x', label=f'{model_type} ID')
    # plt.title(f"max_orig-max_pred {dataset_name}")
    # plt.xlabel("SNR diff", fontsize=12)
    # plt.ylabel("MSE", fontsize=12)
    # plt.ylim(0, 2)
    # plt.tight_layout()
    # plt.legend(loc='upper right', fontsize='small')
    # plt.savefig(f"./paper_figures/{dataset_name}_mse_max_max_snr.png", dpi=300, bbox_inches='tight')
    # plt.close()


    #     # plot peak diff of snr
        c = next(colors)
        plt.scatter(peak_diff_HW, mses, marker='.', color=c, label=f'{model_type} HW')
        plt.scatter(peak_diff_ID, mses, marker='x', color=c, label=f'{model_type} ID')
    plt.title(f"SNR peak diff for {dataset_name}")
    plt.xlabel("SNR diff", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.legend(loc='upper left', fontsize='small')
    plt.savefig(f"./paper_figures/{dataset_name}_mse_peak_snr_more_models.png", dpi=300, bbox_inches='tight')
    plt.close()

    mse_all = np.array([item for sublist in mse_all for item in sublist])
    peak_snr_diff_all_HW = np.array([item for sublist in peak_snr_diff_all_HW for item in sublist])
    peak_snr_diff_all_ID = np.array([item for sublist in peak_snr_diff_all_ID for item in sublist])
    mse_threshold = np.where(mse_all < 0.25)[0]
    pcorr = stats.pearsonr(peak_snr_diff_all_HW[mse_threshold], mse_all[mse_threshold])
    print(f'{dataset_name}\tHW\t{pcorr.statistic}\t{pcorr.pvalue}')
    pcorr = stats.pearsonr(peak_snr_diff_all_ID[mse_threshold], mse_all[mse_threshold])
    print(f'{dataset_name}\tID\t{pcorr.statistic}\t{pcorr.pvalue}')
    # origi = np.argsort(mses)
    # diffHW = np.argsort(peak_diff_HW)
    # diffID = np.argsort(peak_diff_ID)
    # print('sorting analysis with peak diff')
    # print('HW\t\tID')
    # HWs = []
    # IDs = []
    # for i in range(1, len(origi) + 1):
    #     cntHW = 0
    #     cntID = 0
    #     for j in range(0, i):
    #         if origi[j] == diffHW[j]:
    #             cntHW += 1
    #         if origi[j] == diffID[j]:
    #             cntID += 1
    #     print(f'{cntHW}/{i}\t\t{cntID}/{i}')
    #     HWs.append(cntHW)
    #     IDs.append(cntID)
    # plt.clf()
    # plt.plot(np.arange(1,11), HWs, marker='.', color='b', label='HW')
    # plt.plot(np.arange(1,11), IDs, marker='.', color='r', label='ID')
    # # plt.grid(True, which="both", ls="-")
    # plt.title(f"#correct {model_type} {dataset_name}")
    # plt.xlabel("Order ", fontsize=12)
    # plt.ylabel("Correct", fontsize=12)
    # # plt.xticks(range(1, len(ls_sizes) + 1), x_ticks)
    # plt.ylim(0, 10)
    # plt.tight_layout()
    # plt.legend(loc='upper right', fontsize='small')
    # plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_order_correctness_orig_peak_diff.png", dpi=300, bbox_inches='tight')
    # plt.close()
    # #
    # diffHW = np.argsort(snr_orig_pred_HW)
    # diffID = np.argsort(snr_orig_pred_ID)
    # HWs = []
    # IDs = []
    # print('sorting analysis with max-max diff')
    # print('HW\t\tID')
    # for i in range(1, len(origi) + 1):
    #     cntHW = 0
    #     cntID = 0
    #     for j in range(0, i):
    #         if origi[j] == diffHW[j]:
    #             cntHW += 1
    #         if origi[j] == diffID[j]:
    #             cntID += 1
    #     print(f'{cntHW}/{i}\t\t{cntID}/{i}')
    #     HWs.append(cntHW)
    #     IDs.append(cntID)
    # plt.clf()
    # plt.plot(np.arange(1,11), HWs, marker='.', color='b', label='HW')
    # plt.plot(np.arange(1,11), IDs, marker='.', color='r', label='ID')
    # # plt.grid(True, which="both", ls="-")
    # plt.title(f"#correct {model_type} {dataset_name}")
    # plt.xlabel("Order", fontsize=12)
    # plt.ylabel("Correct", fontsize=12)
    # # plt.xticks(range(1, len(ls_sizes) + 1), x_ticks)
    # plt.ylim(0, 10)
    # plt.tight_layout()
    # plt.legend(loc='upper right', fontsize='small')
    # plt.savefig(f"{folder_results}/{dataset_name}_{model_type}_order_correctness_max_max_diff.png", dpi=300, bbox_inches='tight')
    # plt.close()
    #
    #
    #
    #
    # print(mses)
    # print(snr_orig_pred_HW)
    # print(snr_orig_pred_ID)
    # print(snr_orig_encod_HW)
    # print(snr_orig_encod_ID)
