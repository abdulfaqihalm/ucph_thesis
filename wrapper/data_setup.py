from torch.utils.data import Dataset, DataLoader
from torch import is_tensor
import pandas as pd
from wrapper import utils

class SequenceDataset(Dataset):
    def __init__(self, seq_fasta_path: str, prom_seq_fasta_path: str, label_json_path: str, transfrom=None) -> None:
        super().__init__()
        self.label = pd.read_json(label_json_path)
        self.seq = utils.create_seq_tensor(seq_fasta_path)
        self.prom_seq = utils.create_seq_tensor(prom_seq_fasta_path)
    
    def __len__(self) -> int:
        return len(self.label)
    
    def __getitem__(self, idx) -> dict[str, any]:
        if is_tensor(idx):
            idx = idx.tolist()

        return {
            "seq": self.seq[idx],
            "prom_seq": self.prom_seq[idx],
            "meth_case": self.label["meth_case"].iloc[idx],
            "meth_control": self.label["meth_control"].iloc[idx]
        }

class TrainLoader(DataLoader):
    pass

class TestLoader(DataLoader):
    pass

if __name__=="__main__":
    path_to_seq = "/Users/faqih/Documents/UCPH/Thesis/code/data/train_test_data/motif_fasta_test_SPLIT_1.fasta"
    path_to_label = "/Users/faqih/Documents/UCPH/Thesis/code/data/train_test_data/test_label_SPLIT_1.json"
    label = pd.read_json(path_to_label)
    meth_control = label["meth_control"]
    meth_case = label["meth_case"]
    print(label)