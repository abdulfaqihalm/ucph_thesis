from wrapper.utils import plot_correlation
import pandas as pd 
import numpy as np
import torch 
from wrapper import utils

# true_df = pd.read_json("data/train_test_data_500_2/test_meta_data_SPLIT_3.json")
# true_meth_case = true_df["meth_case"]

# # Control trained with Suppl. Data
# pred_meth_case_wo_m6A = pd.read_csv("data/outputs/prediction_500_wo_m6A_control.csv", header=None, names=["pred"])
# # Control_2 trained with Union Data
# pred_meth_case_wo_m6A_2 = pd.read_csv("data/outputs/prediction_500_wo_m6A_control_2.csv", header=None, names=["pred"])
# pred_meth_case = pd.read_csv("data/outputs/prediction_500.csv", header=None, names=["pred"])
# #plot_correlation(true_meth_case.to_numpy(), pred_meth_case_wo_m6A.pred.to_numpy(), ".", "pred_meth_case_wo_m6A")
# plot_correlation(true_meth_case.to_numpy(), pred_meth_case_wo_m6A_2.pred.to_numpy(), ".", "pred_meth_case_wo_m6A_2_union")
# plot_correlation(true_meth_case.to_numpy(), pred_meth_case.pred.to_numpy(), ".", "pred_meth_case")

m6A_data = pd.read_json("data/train_test_data_500_2/test_case_m6A_prob_data_SPLIT_3.json")
# m6A_data_torch = torch.from_numpy(m6A_data["m6A_prob_case"].to_numpy())
m6A_data_torch = torch.from_numpy(np.stack(m6A_data["m6A_prob_case"].to_numpy()))
print(m6A_data_torch.shape)
print(m6A_data_torch[0].shape)
seq = utils.create_seq_tensor("data/train_test_data_500_2/motif_fasta_test_SPLIT_3.fasta")
print(seq.shape)
print(seq[0])
coba = torch.concat((seq[0],m6A_data_torch[0].unsqueeze(0)))
print(coba)
print(coba.shape)
print(coba[4])