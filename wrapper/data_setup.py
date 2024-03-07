from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from wrapper import utils

class SequenceDataset(Dataset):
    def __init__(self, seq_fasta_path: str, prom_seq_fasta_path: str, label_json_path: str, transfrom=None) -> None:
        super().__init__()
        self.label = pd.read_json(label_json_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
    
    def __len__(self) -> int:
        return len(self.label)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Add the methylation level to the site (51th position)
        seq_at = self.seq[idx]
        seq_at[0,seq_at.shape[1]//2] += (self.label["meth_control"].iloc[idx]/100)
        #self.prom_seq = utils.create_seq_tensor(self.prom_seq_fasta_path, idx)
        return {
            "seq": seq_at,
            #"prom_seq": self.prom_seq[idx],
            "meth_case": torch.tensor(self.label["meth_case"].iloc[idx]/100, dtype=torch.float32),
            "meth_control": torch.tensor(self.label["meth_control"].iloc[idx]/100, dtype=torch.float32)
        }


if __name__=="__main__":
    # Example usage from parent dir:  python -m wrapper.data_setup
    path_to_seq = "data/train_test_data/motif_fasta_test_SPLIT_1.fasta"
    path_to_prom = "data/train_test_data/promoter_fasta_test_SPLIT_1.fasta"
    path_to_label = "data/train_test_data/test_label_SPLIT_1.json"
    dataset = SequenceDataset(path_to_seq, path_to_prom, path_to_label) 
    data = dataset[2] # AATGTGGAAATAAGTTGTGTTACTACATGTGTGTAATCCTAGGGTGCAGGACACCGGCCGGGAGGTTCCATAGAGTGATGGGTTCTGCAGGTAACTCATCC
    print(data['seq'])
    print(data['seq'].dtype)
    print(data['seq'].shape)
    print(data['seq'].shape[1]//2)
    print(data['seq'][0,data['seq'].shape[1]//2])
    print(data["meth_case"].dtype)
    print(data["meth_case"].shape)
    print(data["meth_control"].dtype)
    print(data["meth_control"].shape)