from Bio import SeqIO
import numpy as np 
import torch

def one_hot(seq: str) -> np.ndarray:
    """
    One-hot encode a sequence
    """
    look_up_set = set("ACGTN")
    look_up_table = {"A": [1.0, 0.0, 0.0, 0.0],
                     "C": [0.0, 1.0, 0.0, 0.0],
                     "G": [0.0, 0.0, 1.0, 0.0],
                     "T": [0.0, 0.0, 0.0, 1.0],
                     "N": [0.0, 0.0, 0.0, 0.0]}
    
    result = []
    for base in seq: 
        if base not in look_up_set:
            raise ValueError(f"Base {base} is not valid character (ACGTN)")
        
        result.append(look_up_table[base])
    result = np.array(result)
    return result.T

def create_seq_tensor(path_to_fasta: str) -> torch.Tensor:
    """
    Create a tensor of sequences from a fasta file

    param:  path_to_fasta: str: path to the fasta file
    return: torch.Tensor: tensor of sequences
    """
    seq_list = []

    for seq_record in SeqIO.parse(path_to_fasta, format="fasta"):
        seq_list.append(one_hot(str(seq_record.seq)))

    seq_list = torch.Tensor(np.array(seq_list, dtype=np.float32))
    return seq_list
        


if __name__=="__main__":
    path_to_fasta = "/Users/faqih/Documents/UCPH/Thesis/code/data/train_test_data/motif_fasta_test_SPLIT_1.fasta"
    print(one_hot("ACGTN")) 
    # Should return:
    # [[1. 0. 0. 0. 0.]
    # [0. 1. 0. 0. 0.]
    # [0. 0. 1. 0. 0.]
    # [0. 0. 0. 1. 0.]]

    # print(create_seq_tensor(path_to_fasta))