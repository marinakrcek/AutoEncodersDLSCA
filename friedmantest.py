from scipy import stats
import scikit_posthocs as sp
import numpy as np

#
# l20 = [
#     0.1870051622390747,
#     0.29277599994451514,
#     0.28549465662372564,
#     0.1617896407842636,
#     0.1635696291923523,
#     0.29529479730250846
# ]
# l40 = [
#     0.13086526095867157,
#     0.2733460739661241,
#     0.2714263496804373,
#     0.13845962285995483,
#     0.1708582490682602,
#     0.2653043233631831
# ]
# l50 = [
#     0.11311204731464386,
#     0.26966003421722723,
#     0.2631270242504308,
#     0.18124902248382568,
#     0.12778352200984955,
#     0.28344584860946853
# ]
#
# l100 = [
#     0.06588806211948395,
#     0.24084066799682752,
#     0.2398157637273421,
#     0.08382844179868698,
#     0.0861266478896141,
#     0.24118557470089408
# ]
#
# l200 = [
#     0.05722171813249588,
#     0.2108517630500027,
#     0.1986773419688491,
#     0.06086423248052597,
#     0.053578492254018784,
#     0.20121962932149817
# ]
#
# l250 = [
#     0.055480893701314926,
#     0.21122164807288413,
#     0.27160322500974476,
#     0.11811843514442444,
#     0.09448438882827759,
#     0.2588380574171631
# ]
# l400 = [
#     0.031925931572914124,
#     0.1187103447703642,
#     0.20673354305167835,
#     0.06890803575515747,
#     0.08311604708433151,
#     0.1576817746026861
# ]
#
# l500 = [
#     0.05789928510785103,
#     0.14473463857523036,
#     0.2348143345639549,
#     0.07563463598489761,
#     0.09618181735277176,
#     0.2610358233771501
# ]
# print(stats.friedmanchisquare(l20, l40, l50, l100, l200, l250, l400, l500))
#
# #combine three groups into one array
# data = np.array([l20, l40, l50, l100, l200, l250, l400, l500])
#
# #perform Nemenyi post-hoc test
# ress = sp.posthoc_nemenyi_friedman(data.T)
# ress.columns = ["l20", "l40", "l50", "l100", "l200", "l250", "l400", "l500"]
# ress.index = ["l20", "l40", "l50", "l100", "l200", "l250", "l400", "l500"]
# print(ress.to_string())
#
# l20 = [
#     0.28349196501394075,
#     0.18906241655349731,
#     0.1870051622390747,
#     0.29277599994451514,
#     0.28549465662372564,
#     0.1617896407842636,
#     0.1635696291923523,
#     0.29529479730250846
# ]
#
# l40 = [
#     0.26898043869001503,
#     0.1258777678012848,
#     0.13086526095867157,
#     0.2733460739661241,
#     0.2714263496804373,
#     0.13845962285995483,
#     0.1708582490682602,
#     0.2653043233631831
# ]
#
# l50 = [
#     0.26252753080191815,
#     0.1195998415350914,
#     0.11311204731464386,
#     0.26966003421722723,
#     0.2631270242504308,
#     0.18124902248382568,
#     0.12778352200984955,
#     0.28344584860946853
# ]
#
# l100 = [
#     0.2386866015264412,
#     0.07028467953205109,
#     0.06588806211948395,
#     0.24084066799682752,
#     0.2398157637273421,
#     0.08382844179868698,
#     0.0861266478896141,
#     0.24118557470089408
# ]
#
# l200 = [
#     0.1997243355681004,
#     0.04967227950692177,
#     0.05722171813249588,
#     0.2108517630500027,
#     0.1986773419688491,
#     0.06086423248052597,
#     0.053578492254018784,
#     0.20121962932149817
# ]
# l250 = [
#     0.1754606801181369,
#     0.04516558721661568,
#     0.055480893701314926,
#     0.21122164807288413,
#     0.27160322500974476,
#     0.11811843514442444,
#     0.09448438882827759,
#     0.2588380574171631
# ]
#
# l400 = [
#     0.1217923122732464,
#     0.042653992772102356,
#     0.031925931572914124,
#     0.1187103447703642,
#     0.20673354305167835,
#     0.06890803575515747,
#     0.08311604708433151,
#     0.1576817746026861
# ]
#
# print(stats.friedmanchisquare(l20, l40, l50, l100, l200, l250, l400))
#
#
# #combine three groups into one array
# data = np.array([l20, l40, l50, l100, l200, l250, l400])
#
# #perform Nemenyi post-hoc test
# ress = sp.posthoc_nemenyi_friedman(data.T)
# ress.columns = ["l20", "l40", "l50", "l100", "l200", "l250", "l400"]
# ress.index = ["l20", "l40", "l50", "l100", "l200", "l250", "l400"]
# print(ress.to_string())
# #
# mlp_orig = [22, 22, 44, 38]
# mlp_ae_mlp = [22, 28, 35, 47]
# mlp_ae_cnn = [14, 25, 37, 35]
# print(stats.friedmanchisquare(mlp_orig, mlp_ae_mlp, mlp_ae_cnn))
# cnn_orig = [16, 15, 29, 26]
# cnn_ae_mlp = [12, 6, 25, 4]
# cnn_ae_cnn = [10, 20, 18, 25]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
# #
# cnn_orig = [22, 22, 44, 38, 16, 15, 29, 26]
# cnn_ae_mlp = [22, 28, 35, 47, 12, 6, 25, 4]
# cnn_ae_cnn = [14, 25, 37, 35, 10, 20, 18, 25]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
# print(stats.rankdata([cnn_orig, cnn_ae_mlp, cnn_ae_cnn], axis=0))
# import scikit_posthocs as sp
# sp.posthoc_conover_friedman(
# cnn_orig = [22, 22, 16, 15]
# cnn_ae_mlp = [22, 28, 12, 6]
# cnn_ae_cnn = [14, 25, 10, 20]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))


# cnn_orig = [44, 38, 29, 26]
# cnn_ae_mlp = [35, 47, 25, 4]
# cnn_ae_cnn = [37, 35, 18, 25]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))

# cnn_orig = [314185.009, 359752.8, 446422.4286, 357711.2475, 321327.8248, 171559.8372]
# cnn_ae_mlp = [196126.3191, 265892.6176, 361597.2121, 423521.2941, 327517.8919, 142645.7872]
# cnn_ae_cnn = [272359.5443, 383584.9855, 473825.8198, 453262.4762, 370796.5217, 226223.2603]
# print(stats.rankdata([cnn_orig, cnn_ae_mlp, cnn_ae_cnn], axis=0))
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
#
# cnn_orig = [314185.009, 357711.2475]
# cnn_ae_mlp = [196126.3191, 423521.2941]
# cnn_ae_cnn = [272359.5443, 453262.4762]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
#
# cnn_orig = [446422.4286, 171559.8372]
# cnn_ae_mlp = [361597.2121, 142645.7872]
# cnn_ae_cnn = [473825.8198, 226223.2603]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
# #
# #
# # cnn_orig = [1, -2.5, 4.5, 4]
# # cnn_ae_mlp = [2, 12, 1.5, 3.5]
# # cnn_ae_cnn = [-3, 5.5, 7.5, 5]
# #
# # print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
#
# cnn_orig = [48, 0, 100, 4]
# cnn_ae_cnn = [100, 48, 100, 96]
# cnn_ae_mlp = [100, 1, 100, 77]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
# print(stats.ttest_rel(cnn_orig, cnn_ae_mlp))
# print(stats.ttest_rel(cnn_orig, cnn_ae_cnn))
#
# cnn_orig = [48, 0, 100, 4]
# cnn_ae_cnn = [100, 0, 100, 100]
# cnn_ae_mlp = [100, 100, 100, 95]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
# print(stats.ttest_rel(cnn_orig, cnn_ae_mlp))
# print(stats.ttest_rel(cnn_orig, cnn_ae_cnn))
#
# cnn_orig = [25, 95, 100, 99]
# cnn_ae_cnn = [9, 100, 100, 100]
# cnn_ae_mlp = [0, 0, 99, 4]
#
# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
# stats.ttest_ind(a=cnn_orig, b=cnn_ae_cnn, equal_var=True)
#
# cnn_orig = [25, 95, 100, 99]
# cnn_ae_cnn = [75, 100, 100, 98]
# cnn_ae_mlp = [66, 0, 100, 0]

# print(stats.friedmanchisquare(cnn_orig, cnn_ae_mlp, cnn_ae_cnn))
# print(stats.ttest_rel(cnn_orig, cnn_ae_mlp))
# print(stats.ttest_rel(cnn_orig, cnn_ae_cnn))
# stats.ttest_ind(a=cnn_orig, b=cnn_ae_cnn, equal_var=True)


# cnn_orig = [48, 100, 0, 4, 25, 100, 95, 99]
# cnn_ae_cnn = [100, 100, 48, 96, 9, 100, 100, 100]
# cnn_ae_mlp = [100, 100, 1, 77, 0, 99, 0, 4]
#
#
# print(stats.ttest_rel(cnn_orig, cnn_ae_cnn))
# print(stats.ttest_rel(cnn_orig, cnn_ae_mlp))
#
# cnn_orig2 = [48, 100, 0, 4, 25, 100, 95, 99]
# cnn_ae_cnn2 = [100, 100, 0, 100, 75, 100, 100, 98]
# cnn_ae_mlp2 = [100, 100, 100, 95, 66, 100, 0, 0]
#
# print(stats.ttest_rel(cnn_orig2, cnn_ae_cnn2))
# print(stats.ttest_rel(cnn_orig2, cnn_ae_mlp2))
#
# print(stats.ttest_rel(cnn_ae_cnn, cnn_ae_cnn2))
# print(stats.ttest_rel(cnn_ae_mlp, cnn_ae_mlp2))
#
# print('per dataset')
# # print(stats.ttest_rel(cnn_orig[:4], cnn_ae_cnn[:4]))
# # print(stats.ttest_rel(cnn_orig[:4], cnn_ae_mlp[:4]))
# print(stats.ttest_rel(cnn_orig[4:], cnn_ae_cnn[4:]))
# print(stats.ttest_rel(cnn_orig[4:], cnn_ae_mlp[4:]))
# print(stats.ttest_rel(cnn_ae_cnn[:4], cnn_ae_cnn2[:4]))
# print(stats.ttest_rel(cnn_ae_mlp[:4], cnn_ae_mlp2[:4]))
# # print(stats.ttest_rel(cnn_ae_cnn[4:], cnn_ae_cnn2[4:]))
# # print(stats.ttest_rel(cnn_ae_mlp[4:], cnn_ae_mlp2[4:]))
# print('per dataset, with stand.')
# # print(stats.ttest_rel(cnn_orig2[:4], cnn_ae_cnn2[:4]))
# # print(stats.ttest_rel(cnn_orig2[:4], cnn_ae_mlp2[:4]))
# print(stats.ttest_rel(cnn_orig2[4:], cnn_ae_cnn2[4:]))
# print(stats.ttest_rel(cnn_orig2[4:], cnn_ae_mlp2[4:]))

mingewost1 = [1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
mingewst1 = [23.9,  34, 57.95, 53.65, 77.6, 1, 1, 1, 25.95, 1, 1, 1, 1, 1, 1]
print(stats.ttest_rel(mingewost1, mingewst1))
print(stats.ttest_ind(mingewost1, mingewst1))

mingewost2 = [67.8, 115.75, 1.65, 1.15, 1, 3.8, 1, 1, 18.5, 1.95, 1.8, 2.55, 1, 7.95, 1, 1.55]
mingewst2 = [28.35, 77.9, 44.2, 86.4, 72.1, 136.7, 1, 1, 21.85, 28.95, 6.65, 21.15, 46.1, 48.75, 1.2, 6.3]
print(stats.ttest_rel(mingewost2, mingewst2))
print(stats.ttest_ind(mingewost2, mingewst2))
# print(stats.ttest_rel(mingewost1, mingewost2))
# print(stats.ttest_ind(mingewost1, mingewost2))
# print(stats.ttest_rel(mingewst1, mingewst2))
# print(stats.ttest_ind(mingewst1, mingewst2))
#
# sth1 = [24, 5, 0, 0, 0, 0, 5, 1]
# sth2 = [69, 100, 0, 51, 37, 100, 2, 17]
# print(stats.ttest_ind(sth1, sth2))
# print(stats.ttest_rel(sth1, sth2))
# print(stats.ranksums(sth1, sth2, alternative='less'))
#
# sth1 = [100, 65, 97, 100, 69, 100, 0, 51, 100, 79, 97, 98, 35, 97, 0, 16]
# sth2 = [51, 100, 73, 99, 37, 100, 2, 17, 19, 99, 89, 100, 17, 30, 10, 66]
# print(stats.ttest_ind(sth1, sth2))
# print(stats.ttest_rel(sth1, sth2))
# print(stats.ranksums(sth1, sth2, alternative='less'))


# sth1 = [100, 65, 97, 100, 69, 100, 0, 51]
# sth2 = [51, 100, 73, 99, 37, 100, 2, 17]
# print(stats.ttest_ind(sth1, sth2))
# print(stats.ttest_rel(sth1, sth2))

# sth1 = [100, 79, 97, 98, 35, 97, 0, 16]
# sth2 = [19, 99, 89, 100, 17, 30, 10, 66]
# print(stats.ttest_ind(sth1, sth2))
# print(stats.ttest_rel(sth1, sth2))

# sth1 = [100, 65, 97, 100, 69, 100, 0, 51]
# sth2 = [100, 79, 97, 98, 35, 97, 0, 16]
# print(stats.ttest_ind(sth1, sth2))
# print(stats.ttest_rel(sth1, sth2))

# sth1 = [51, 100, 73, 99, 37, 100, 2, 17]
# sth2 = [19, 99, 89, 100, 17, 30, 10, 66]
# print(stats.ttest_ind(sth1, sth2))
# print(stats.ttest_rel(sth1, sth2))
#
# sth1 = [100, 65, 97, 100, 69, 100, 0, 51, 51, 100, 73, 99, 37, 100, 2, 17]
# sth2 = [100, 79, 97, 98, 35, 97, 0, 16, 19, 99, 89, 100, 17, 30, 10, 66]
# print(stats.ttest_ind(sth1, sth2))
# print(stats.ttest_rel(sth1, sth2))
# print(stats.ranksums(sth1, sth2))
# print(stats.mannwhitneyu(sth1, sth2, ))
