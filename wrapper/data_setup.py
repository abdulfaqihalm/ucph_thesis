from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from wrapper import utils

class SequenceDataset(Dataset):
    def __init__(self, seq_fasta_path: str, prom_seq_fasta_path: str, meta_data_path: str, m6A_mode: str="no", m6A_info_path: None|str =None, transfrom=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
        self.meth_case = (self.meta_data["meth_case"] if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"]*100)
        self.meth_control = (self.meta_data["meth_control"] if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"]*100)
        self.m6A_info = pd.DataFrame([])
        if m6A_info_path!="no":
            self.m6A_info = pd.read_json(m6A_info_path)
            
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if not (self.m6A_info.empty):
            m6A = self.m6A_info.iloc[idx, 0]
            m6A = torch.tensor(m6A, dtype=torch.float32).unsqueeze(0)
            seq = torch.cat((self.seq[idx],m6A), dim=0)
        else:
            seq = self.seq[idx]
        
        # Add the methylation level to the site (51th position)
        #seq_at[0,seq_at.shape[1]//2] += (self.label["meth_control"].iloc[idx]/100)
        #self.prom_seq = utils.create_seq_tensor(self.prom_seq_fasta_path, idx)
        return {
            "seq": seq,
            #"prom_seq": self.prom_seq[idx],
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32)
        }

class TrainDataset(Dataset):
    def __init__(self, seq_fasta_path: str, m6A_list_path: str, meta_data_path: str, transfrom=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
        self.meth_case = (self.meta_data["meth_case"]/100 if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"])
        self.meth_control = (self.meta_data["meth_control"]/100 if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"])
        self.m6A_list = torch.from_numpy(np.stack((pd.read_json(m6A_list_path)).iloc[:,0].to_numpy()))
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Add the methylation level to the site (51th position)
        #seq_at[0,seq_at.shape[1]//2] += (self.label["meth_control"].iloc[idx]/100)
        #self.prom_seq = utils.create_seq_tensor(self.prom_seq_fasta_path, idx)
        return {
            "seq": torch.concat((self.seq[idx],self.m6A_list[idx].unsqueeze(0))),   
            #"prom_seq": self.prom_seq[idx],
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32)
        }


class TestDataset(Dataset):
    def __init__(self, seq_fasta_path: str, meta_data_path: str, transfrom=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
        self.meth_case = (self.meta_data["meth_case"]/100 if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"])
        self.meth_control = (self.meta_data["meth_control"]/100 if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"])
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Add the methylation level to the site (51th position)
        #seq_at[0,seq_at.shape[1]//2] += (self.label["meth_control"].iloc[idx]/100)
        #self.prom_seq = utils.create_seq_tensor(self.prom_seq_fasta_path, idx)
        return {
            "seq": torch.concat((self.seq[idx],self.m6A_list[idx].unsqueeze(0))),   
            #"prom_seq": self.prom_seq[idx],
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32)
        }

class SequenceInferenceDataset(Dataset):
    def __init__(self, seq_fasta_path) -> None:
        super().__init__()
        self.seq = utils.create_seq_tensor(seq_fasta_path)
    
    def __len__(self) -> int:
        return self.seq.shape[0]
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_at = self.seq[idx]
        return seq_at


if __name__=="__main__":
    # Example usage from parent dir:  python -m wrapper.data_setup
    path_to_seq = "data/train_test_data_500_2/motif_fasta_test_SPLIT_1.fasta"
    path_to_prom = "data/train_test_data_500_2/promoter_fasta_test_SPLIT_1.fasta"
    path_to_label = "data/train_test_data_500_2/test_meta_data_SPLIT_1.json"
    path_to_m6A_info = "data/train_test_data_500_2/test_case_m6A_prob_data_SPLIT_1.json"
    #dataset = SequenceDataset(path_to_seq, path_to_prom, path_to_label) 
    dataset = SequenceDataset(path_to_seq, path_to_prom, path_to_label, path_to_m6A_info) 
    m6A_info = pd.read_json(path_to_m6A_info)
    print(m6A_info)
    data = dataset[2] # AGC....CCTC
    print(data['seq'])
    print(data['seq'].dtype)
    print(data['seq'].shape)
    # print(data['seq'].shape[1]//2)
    print(data['seq'][0,data['seq'].shape[1]//2])
    print(data['seq'][4,data['seq'].shape[1]//2])
    # print(data["meth_case"].dtype)
    # print(data["meth_case"].shape)
    # print(data["meth_control"].dtype)
    # print(data["meth_control"].shape)
    # # coba_data = pd.read_csv("data/train_test_data_2/motif_fasta_test_SPLIT_1.fasta")

    # print((utils.create_seq_tensor(path_to_seq)).shape)

    # path_to_seq = "data/train_test_data_500_2/motif_fasta_test_SPLIT_1.fasta"
    # path_to_m6A_control = "data/train_test_data_500_2/test_case_m6A_prob_data_SPLIT_1.json"
    # path_to_meta_data = "data/train_test_data_500_2/test_meta_data_SPLIT_1.json"
    # dataset = TrainDataset(path_to_seq, path_to_m6A_control, path_to_meta_data)
    # print(len(dataset))
    # data = dataset[2]
    # print(data["seq"].shape)
    # print(data["seq"])
    # print(f"{data["meth_case"]}, {data["seq"][4,500]}")