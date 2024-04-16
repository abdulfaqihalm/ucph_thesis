from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import h5py 
from wrapper import utils

class SequenceDataset(Dataset):
    def __init__(self, seq_fasta_path: str, meta_data_path: str,  prom_seq_fasta_path: None|str=None, m6A_info: None|str = "no", m6A_info_path: None|str =None, target: str="case" , transform: str="one-hot", path_to_embedding: str|None=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
        self.target=target
        self.prom_seq = None
        if prom_seq_fasta_path:
            self.prom_seq = utils.create_seq_tensor(prom_seq_fasta_path)

        self.meth_case = (self.meta_data["meth_case"]/100 if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"])
        self.meth_control = (self.meta_data["meth_control"]/100 if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"])
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
    def __init__(self, seq_fasta_path: str, meta_data_path: str,  prom_seq_fasta_path: None|str=None, m6A_info: None|str = "no", m6A_info_path: None|str =None, transform: str="one-hot", path_to_embedding: str|None=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.meth_case = (self.meta_data["meth_case"]/100 if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"])
        self.meth_control = (self.meta_data["meth_control"]/100 if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"])
        self.meth_case_weight = utils.calculate_weights(self.meth_case.to_numpy())
        self.meth_control_weight = utils.calculate_weights(self.meth_control.to_numpy())

        self.transform = transform
        if self.transform == "one-hot":
            self.seq = utils.create_seq_tensor(seq_fasta_path)
            self.prom_seq = None
            if prom_seq_fasta_path:
                self.prom_seq = utils.create_seq_tensor(prom_seq_fasta_path)

            self.m6A_info = m6A_info
            self.m6A_df = pd.DataFrame([])
            if m6A_info not in ["no", "add_prob_middle", "add_flag_middle"] and m6A_info_path is not None:
                self.m6A_df = pd.read_json(m6A_info_path)
            elif m6A_info in ["no", "add_prob_middle", "add_flag_middle"] and m6A_info_path is not None:
                raise ValueError(f"m6A_info: {m6A_info} or m6A_info_path: {m6A_info_path} is not valid")
        elif self.transform == "gene2vec":
            if path_to_embedding is None:
                raise ValueError("Path to embedding file is required for gene2vec transformation")
            self.seq = utils.create_seq_tensor(seq_fasta_path, transform=self.transform, path_to_embedding=path_to_embedding)
            
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform == "one-hot":
            if not (self.m6A_df.empty):
                m6A = self.m6A_df.iloc[idx, 0]
                m6A = torch.tensor(m6A, dtype=torch.float32).unsqueeze(0)
                seq = torch.cat((self.seq[idx],m6A), dim=0)
            else:
                seq = self.seq[idx]
            if self.prom_seq is not None:
                prom_seq = self.prom_seq[idx]
                if prom_seq.shape[0] < seq.shape[0]:
                    # Add 5th channel as zeros for prom-seq
                    prom_seq = torch.cat((prom_seq, torch.zeros(seq.shape[0]-prom_seq.shape[0], prom_seq.shape[1])), dim=0)
                if prom_seq.shape[0] > seq.shape[0]:
                    raise ValueError(f"Promoter channel number is greater than the motif sequence channel: {prom_seq.shape[0]} > {seq.shape[0]}")
                seq = torch.cat((prom_seq, seq), dim=1)
        elif self.transform == "gene2vec":
            seq = self.seq[idx]
        
        return {
            "seq": seq,
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_case_weight": torch.tensor(self.meth_case_weight[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32),
            "meth_control_weight": torch.tensor(self.meth_control_weight[idx], dtype=torch.float32)
        }
    





class SequenceDatasetDualFilter(Dataset):
    def __init__(self, seq_fasta_path: str, meta_data_path: str,  prom_seq_fasta_path: None|str=None, m6A_info: None|str = "no", m6A_info_path: None|str =None, transform: str="one-hot", path_to_embedding: str|None=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)

        # Additional filter
        filter_series = ((self.meta_data["meth_case"] >=0.15) & (self.meta_data["meth_case"] <=0.40) | (self.meta_data["meth_case"] >=0.95) & (self.meta_data["meth_case"] <=1.0)) & ((self.meta_data["meth_control"] >=0.15) & (self.meta_data["meth_control"] <=0.40) | (self.meta_data["meth_control"] >=0.95) & (self.meta_data["meth_control"] <=1.0))
        self.meta_data = self.meta_data[filter_series]

        self.meth_case = (self.meta_data["meth_case"]/100 if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"])
        self.meth_control = (self.meta_data["meth_control"]/100 if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"])
        self.meth_case_weight = utils.calculate_weights(self.meth_case.to_numpy())
        self.meth_control_weight = utils.calculate_weights(self.meth_control.to_numpy())

        self.transform = transform
        if self.transform == "one-hot":
            self.seq = utils.create_seq_tensor(seq_fasta_path)
            self.seq = self.seq[filter_series]
            
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.seq[idx]
        
        return {
            "seq": seq,
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_case_weight": torch.tensor(self.meth_case_weight[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32),
            "meth_control_weight": torch.tensor(self.meth_control_weight[idx], dtype=torch.float32)
        }


class SequenceDatasetDualShortenedFeatures(Dataset):
    def __init__(self, seq_fasta_path: str, meta_data_path: str,  prom_seq_fasta_path: None|str=None, m6A_info: None|str = "no", m6A_info_path: None|str =None, transform: str="one-hot", path_to_embedding: str|None=None) -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.meth_case = (self.meta_data["meth_case"]/100 if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"])
        self.meth_control = (self.meta_data["meth_control"]/100 if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"])
        self.meth_case_weight = utils.calculate_weights(self.meth_case.to_numpy())
        self.meth_control_weight = utils.calculate_weights(self.meth_control.to_numpy())

        self.transform = transform
        if self.transform == "one-hot":
            self.seq = utils.create_seq_tensor(seq_fasta_path)
            self.seq = self.seq[:, :, 251:752]
            
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.seq[idx]
        
        return {
            "seq": seq,
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_case_weight": torch.tensor(self.meth_case_weight[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32),
            "meth_control_weight": torch.tensor(self.meth_control_weight[idx], dtype=torch.float32)
        }


class SequenceDatasetDualGene2Vec(Dataset):
    def __init__(self, file_name: str, dataset: str,  meta_data_path: str,  transform: str="gene2vec") -> None:
        super().__init__()
        self.meta_data = pd.read_json(meta_data_path)
        self.meth_case = (self.meta_data["meth_case"]/100 if self.meta_data["meth_case"].max() > 1 else self.meta_data["meth_case"])
        self.meth_control = (self.meta_data["meth_control"]/100 if self.meta_data["meth_control"].max() > 1 else self.meta_data["meth_control"])
        self.transform = transform
        self.file_name = file_name
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        with h5py.File(self.file_name, "r") as file:
            seq = torch.from_numpy(file[self.dataset][idx:idx+1]).squeeze(0).clone()
        
        return {
            "seq": seq,
            "meth_case": torch.tensor(self.meth_case.iloc[idx], dtype=torch.float32),
            "meth_control": torch.tensor(self.meth_control.iloc[idx], dtype=torch.float32)
        }


if __name__=="__main__":
    # Example usage from parent dir:  python -m wrapper.data_setup
    path_to_seq = "data/dual_outputs/motif_fasta_train_SPLIT_1.fasta"
    # path_to_prom = None
    path_to_prom = "data/dual_outputs/promoter_fasta_test_SPLIT_1.fasta"
    path_to_meta_data = "data/dual_outputs/train_meta_data_SPLIT_1.json"
    # path_to_m6A_info = None
    path_to_m6A_info = "data/single_model/train_control_m6A_flag_data_SPLIT_1.json"

    # m6A_info ["flag_channel", "level_channel", "add_middle", "no"]
    dataset = SequenceDataset(seq_fasta_path=path_to_seq, meta_data_path=path_to_meta_data, prom_seq_fasta_path=None, m6A_info="no", m6A_info_path=None)
    # dataset = SequenceDataset(seq_fasta_path=path_to_seq, meta_data_path=path_to_meta_data, prom_seq_fasta_path=path_to_prom, m6A_info_path=path_to_m6A_info)
    if path_to_m6A_info:
        m6A_info = pd.read_json(path_to_m6A_info)
        print(m6A_info)
    print(dataset.meta_data)
    data = dataset[2] # TTACCAAAAGTACTTTGGAAACTATTCTTAGGCAGATTTACTGTAGACAAATTATTTTTGAAATAATG...NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    # tensor([[0., 0., 1.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [1., 1., 0.,  ..., 0., 0., 0.]])
    print(data["seq"])
    from wrapper.utils import one_hot, one_hot_to_sequence
    print(data["seq"].unsqueeze(0).transpose(2,1).shape)
    print(data["seq"].unsqueeze(0).transpose(2,1))
    temp = data["seq"].unsqueeze(0).transpose(2,1).squeeze(0)
    print(temp)
    print(one_hot_to_sequence(temp)) # harus 2D
    print(data["seq"].dtype)
    print(data["seq"].shape) #[bs, channel, seq_length]
    print(data["seq"][0,500 if path_to_prom else 0 + data["seq"].shape[1]//2])
    if data["seq"].shape[0]==5:
        print(data["seq"][4,500 if path_to_prom else 0 + data["seq"].shape[1]//2])
    print(data["meth_case"])
    # np.savetxt("promoter_with_m6A_channel.txt", data["seq"][4,].detach().numpy())


    # path_to_seq = "data/dual_outputs/motif_fasta_train_SPLIT_1.fasta"
    # # # path_to_prom = None
    # # path_to_prom = "data/dual_outputs/promoter_fasta_test_SPLIT_1.fasta"
    # # path_to_meta_data = "data/dual_outputs/train_meta_data_SPLIT_1.json"
    # # path_to_embedding = "data/embeddings/gene2vec/double_outputs/split_1.model"
    # hdf_file = "/binf-isilon/renniegrp/vpx267/ucph_thesis/data/dual_outputs/hdf5/gene2vec.hdf5"
    # # path_to_prom = None
    # path_to_prom = "data/dual_outputs/promoter_fasta_test_SPLIT_1.fasta"
    # path_to_meta_data = "data/dual_outputs/train_meta_data_SPLIT_1.json"
    # path_to_embedding = "data/embeddings/gene2vec/dual_outputs/split_1.model"
    # from .utils import create_seq_tensor
    # seq = create_seq_tensor(path_to_seq, 2, "gene2vec", path_to_embedding)
    # print(seq)
    # print(seq.dtype)
    # print(seq.shape)
    # dataset = SequenceDatasetDualGene2Vec(hdf_file, dataset="train/motif_SPLIT_1",  meta_data_path=path_to_meta_data)
    # # print(dataset.seq.shape) # [bs, channel, seq_length] torch.Size([103855, 300, 999])
    # data = dataset[2] # AGC....CCTC
    # print(data["seq"])
    # print(data["seq"].dtype)
    # print(data["seq"].shape) #[300, 999]
    # print(data["meth_case"])
    