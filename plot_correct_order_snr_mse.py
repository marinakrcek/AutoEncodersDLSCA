import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict
import matplotlib.cm as cm

model_types = ["ae_mlp", "ae_mlp_dcr", "ae_cnn"]
dataset_names = ["ascad-variable", "dpa_v42"]
leakage_model = "HW"
hiding = ""
nb_runs = 20

# plt.grid(True, which="both", ls="-")
# styles = iter(['-', 'dotted'])
for dataset_name in dataset_names:
    colors = iter(['b', 'r', 'g'])
    linewidths = iter([5, 3, 1])
    # lstyle = next(styles)
    figure = plt.gcf()
    for model_type in model_types:
        folder_results = f"./{model_type}_snr"
        mses = []
        files = []
        snr_orig_pred_HW = []
        snr_orig_pred_ID = []
        snr_orig_encod_HW = []
        snr_orig_encod_ID = []
        peak_diff_HW = []
        peak_diff_ID = []
        # median_se = []
        for file_id in range(0, nb_runs):
            filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_{file_id + 1}.npz"
            npz_file = np.load(filepath, allow_pickle=True)
            mse_ds = npz_file["mse_ds"].item()
            mses.append(mse_ds)
            files.append(filepath)
            # median_se.append(np.mean(np.sqrt(npz_file['mse'])))
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

        best = np.min(mses)
        best_i = np.argmin(mses)
        file = files[best_i]
        print("best AE", file)
        best_AE = np.load(file, allow_pickle=True)
        print(best_AE['hp_values'].item(), best_AE['mse_ds'].item())


        figure.set_size_inches(5, 4)
        origi = np.argsort(mses)
        # origi = np.argsort(median_se)
        # sorted_mses = np.array(mses)[origi]
        # below_mses = np.where(sorted_mses < 0.25)
        # origi = origi[below_mses]
        diffHW = np.argsort(peak_diff_HW)
        diffID = np.argsort(peak_diff_ID)
        print('sorting analysis with peak diff')
        print('HW\t\tID')
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
            print(f'{cntHW}/{i}\t\t{cntID}/{i}')
            HWs.append(cntHW)
            IDs.append(cntID)
        c=next(colors)
        lw = next(linewidths)
        if origi.size == 0:
            continue
        plt.plot(np.arange(1,len(origi)+1), HWs, marker='.', color=c, linewidth=lw, markersize=lw*2, label=f'{model_type} HW')
        plt.plot(np.arange(1,len(origi)+1), IDs, marker='x', color=c, linewidth=lw, markersize=lw*2, label=f'{model_type} ID')
    plt.grid(True, which="both", ls="-")
    plt.title(f"# correct order {dataset_name}")
    plt.xlabel("Order", fontsize=12)
    plt.ylabel("# correct", fontsize=12)
    plt.ylim(0, 10)
    # plt.xlim(1, 20)
    plt.xticks(np.arange(1, nb_runs+1))
    plt.tight_layout()
    plt.legend(fontsize='x-small')  #loc='upper right',
    plt.savefig(f"./paper_figures/{dataset_name}_order_correctness_orig_peak_diff.png", dpi=300, bbox_inches='tight')
    plt.close()

    #     diffHW = np.argsort(snr_orig_pred_HW)
    #     diffID = np.argsort(snr_orig_pred_ID)
    #     HWs = []
    #     IDs = []
    #     print('sorting analysis with max-max diff')
    #     print('HW\t\tID')
    #     for i in range(1, len(origi) + 1):
    #         cntHW = 0
    #         cntID = 0
    #         for j in range(0, i):
    #             if origi[j] == diffHW[j]:
    #                 cntHW += 1
    #             if origi[j] == diffID[j]:
    #                 cntID += 1
    #         print(f'{cntHW}/{i}\t\t{cntID}/{i}')
    #         HWs.append(cntHW)
    #         IDs.append(cntID)
    #     c = next(colors)
    #     lw = next(linewidths)
    #     plt.plot(np.arange(1,len(origi)+1), HWs, marker='.', color=c, label=f'{model_type} HW')
    #     plt.plot(np.arange(1,len(origi)+1), IDs, marker='x', color=c, label=f'{model_type} ID')
    # plt.grid(True, which="both", ls="-")
    # plt.title(f"# correct order {dataset_name}")
    # plt.xlabel("Order", fontsize=12)
    # plt.ylabel("# correct", fontsize=12)
    # plt.ylim(0, 10)
    # plt.tight_layout()
    # plt.legend(fontsize='x-small')  #loc='upper right',
    # plt.savefig(f"./paper_figures/{dataset_name}_order_correctness_max_max_diff.png", dpi=300, bbox_inches='tight')
    # plt.close()

        #
        #
        #
        # print(mses)
        # print(snr_orig_pred_HW)
        # print(snr_orig_pred_ID)
        # print(snr_orig_encod_HW)
        # print(snr_orig_encod_ID)
