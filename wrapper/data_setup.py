from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from wrapper import utils

class SequenceDataset(Dataset):
    def __init__(self, seq_fasta_path: str, meta_data_path: str,  prom_seq_fasta_path: None|str=None, m6A_info: None|str = "no", m6A_info_path: None|str =None, target: str="case" , transfrom=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
        self.target=target
        self.prom_seq = None
        if prom_seq_fasta_path:
            self.prom_seq = utils.create_seq_tensor(seq_fasta_path)

        self.meth_case = (self.meta_data["meth_case"] if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"]*100)
        self.meth_control = (self.meta_data["meth_control"] if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"]*100)
        self.m6A_info = m6A_info
        self.m6A_df = pd.DataFrame([])

        if m6A_info not in ["no", "add_prob_middle", "add_flag_middle"] and m6A_info_path is not None:
            self.m6A_df = pd.read_json(m6A_info_path)
        elif m6A_info in ["no", "add_prob_middle", "add_flag_middle"] and m6A_info_path is not None:
            raise ValueError(f"m6A_info: {m6A_info} or m6A_info_path: {m6A_info_path} is not valid")
            
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #torch.Size([data_channel, seq_length])
        if not (self.m6A_df.empty):
            m6A = self.m6A_df.iloc[idx, 0]
            m6A = torch.tensor(m6A, dtype=torch.float32).unsqueeze(0)
            seq = torch.cat((self.seq[idx],m6A), dim=0)
        else:
            seq = self.seq[idx]
        
        if self.m6A_info == "add_flag_middle":
            seq[0,seq.shape[1]//2] += 1
        elif self.m6A_info == "add_prob_middle":
            meth_lavel = self.meta_data[f"meth_{self.target}"].iloc[idx]
            seq[0,seq.shape[1]//2] += meth_lavel if meth_lavel <= 1 else meth_lavel/100
        
        if self.prom_seq is not None:
            prom_seq = self.prom_seq[idx]
            if prom_seq.shape[0] < seq.shape[0]:
                #prom.Size([4, seq_length]) , seq.Size([5, seq_length])
                prom_seq = torch.cat((prom_seq, torch.zeros(seq.shape[0]-prom_seq.shape[0], prom_seq.shape[1])), dim=0)
                # print(f"prom_seq shape: {prom_seq.shape}")
            if prom_seq.shape[0] > seq.shape[0]:
                raise ValueError(f"Promoter channel number is greater than the motif sequence channel: {prom_seq.shape[0]} > {seq.shape[0]}")
            seq = torch.cat((prom_seq, seq), dim=1)
        
        # Add the methylation level to the site (51th position)
        #seq_at[0,seq_at.shape[1]//2] += (self.label["meth_control"].iloc[idx]/100)
        #self.prom_seq = utils.create_seq_tensor(self.prom_seq_fasta_path, idx)
        return {
            "seq": seq,
            #"prom_seq": self.prom_seq[idx],
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32)
        }

class SequenceDatasetDual(Dataset):
    def __init__(self, seq_fasta_path: str, meta_data_path: str,  prom_seq_fasta_path: None|str=None, m6A_info: None|str = "no", m6A_info_path: None|str =None, transfrom=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
        self.prom_seq = None
        if prom_seq_fasta_path:
            self.prom_seq = utils.create_seq_tensor(seq_fasta_path)

        self.meth_case = (self.meta_data["meth_case"] if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"]*100)
        self.meth_control = (self.meta_data["meth_control"] if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"]*100)
        self.m6A_info = m6A_info
        self.m6A_df = pd.DataFrame([])

        if m6A_info not in ["no", "add_prob_middle", "add_flag_middle"] and m6A_info_path is not None:
            self.m6A_df = pd.read_json(m6A_info_path)
        elif m6A_info in ["no", "add_prob_middle", "add_flag_middle"] and m6A_info_path is not None:
            raise ValueError(f"m6A_info: {m6A_info} or m6A_info_path: {m6A_info_path} is not valid")
            
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #torch.Size([data_channel, seq_length])
        if not (self.m6A_df.empty):
            m6A = self.m6A_df.iloc[idx, 0]
            m6A = torch.tensor(m6A, dtype=torch.float32).unsqueeze(0)
            seq = torch.cat((self.seq[idx],m6A), dim=0)
        else:
            seq = self.seq[idx]
        
        # if self.m6A_info == "add_flag_middle":
        #     seq[0,seq.shape[1]//2] += 1
        # elif self.m6A_info == "add_prob_middle":
        #     meth_lavel = self.meta_data[f"meth_{self.target}"].iloc[idx]
        #     seq[0,seq.shape[1]//2] += meth_lavel if meth_lavel <= 1 else meth_lavel/100
        
        if self.prom_seq is not None:
            prom_seq = self.prom_seq[idx]
            if prom_seq.shape[0] < seq.shape[0]:
                #prom.Size([4, seq_length]) , seq.Size([5, seq_length])
                prom_seq = torch.cat((prom_seq, torch.zeros(seq.shape[0]-prom_seq.shape[0], prom_seq.shape[1])), dim=0)
                # print(f"prom_seq shape: {prom_seq.shape}")
            if prom_seq.shape[0] > seq.shape[0]:
                raise ValueError(f"Promoter channel number is greater than the motif sequence channel: {prom_seq.shape[0]} > {seq.shape[0]}")
            seq = torch.cat((prom_seq, seq), dim=1)
        
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
    path_to_seq = "data/single_model/control/motif_fasta_train_SPLIT_1.fasta"
    # path_to_prom = None
    path_to_prom = "data/train_test_data_500_2/promoter_fasta_test_SPLIT_1.fasta"
    path_to_meta_data = "data/single_model/control/train_meta_data_SPLIT_1.json"
    # path_to_m6A_info = None
    path_to_m6A_info = "data/single_model/control/train_control_m6A_flag_data_SPLIT_1.json"

    # m6A_info ['flag_channel', 'level_channel', 'add_middle', 'no']
    dataset = SequenceDataset(seq_fasta_path=path_to_seq, meta_data_path=path_to_meta_data, prom_seq_fasta_path=path_to_prom, m6A_info="flag_channel", m6A_info_path=path_to_m6A_info)
    # dataset = SequenceDataset(seq_fasta_path=path_to_seq, meta_data_path=path_to_meta_data, prom_seq_fasta_path=path_to_prom, m6A_info_path=path_to_m6A_info)
    if path_to_m6A_info:
        m6A_info = pd.read_json(path_to_m6A_info)
        print(m6A_info)
    data = dataset[2] # AGC....CCTC
    print(data['seq'])
    print(data['seq'].dtype)
    print(data['seq'].shape)
    print(data['seq'][0,500 if path_to_prom else 0 + data['seq'].shape[1]//2])
    if data['seq'].shape[0]==5:
        print(data['seq'][4,500 if path_to_prom else 0 + data['seq'].shape[1]//2])
    print(data['meth_case'])
    np.savetxt("promoter_with_m6A_channel.txt", data['seq'][4,].detach().numpy())
    #print(data['seq'][0,].detach().numpy().tolist())
    # print(data["meth_case"].dtype)
    # print(data["meth_case"].shape)
    # print(data["meth_control"].dtype)
    # print(data["meth_control"].shape)