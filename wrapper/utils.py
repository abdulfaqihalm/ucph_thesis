from Bio import SeqIO
import numpy as np 
import torch
from matplotlib import pyplot as plt 
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

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

def create_seq_tensor(path_to_fasta: str, idx: int|None=None) -> torch.Tensor:
    """
    Create a tensor of sequences from a fasta file

    param:  path_to_fasta: str: path to the fasta file
    param:  idx: int: index (row) of the sequence on the fasta file
    return: torch.Tensor: tensor of sequences
    """
    if idx is None:
        result = []
        for seq_record in SeqIO.parse(path_to_fasta, format="fasta"):
            result.append(one_hot(str(seq_record.seq)))
    else:
        seq_record = SeqIO.parse(path_to_fasta, format="fasta")
        result = one_hot(str(list(seq_record)[idx].seq))
    
    result = torch.from_numpy(np.array(result, dtype=np.float32))
    return result

def plot_loss_function(result_path:str, output_path:str, output_name:str) -> None:
    """
    Plot loss function from a csv file

    param: result_path: str: path to the csv file
    param: output_path: str: path to save the plot
    param: output_name: str: name of the plot
    return: None
    """
    data = pd.read_csv(result_path)
    
    plt.clf()
    plt.plot(data["epoch"], data["train_loss"], label="Train Loss", color="blue")
    plt.plot(data["epoch"], data["val_loss"], label="Validation Loss", color="orange", linestyle="dashed")
    plt.title("Loss Function")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_path}/{output_name}.png")


def plot_correlation(y_true:np.ndarray, y_pred:np.ndarray, output_path:str, output_name:str) -> None:
    """
    Plot correlation

    param: x: np.ndarray: x-axis
    param: y: np.ndarray: y-axis
    param: output_path: str: path to save the plot
    param: output_name: str: name of the plot
    return: None
    """     
    # Get rid future warnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Clear the figure
    plt.clf()
    # Create a jointplot with scatterplot and histograms
    if np.max(y_true) > 1 and np.max(y_pred) > 1:
        max = 100
    else:
        max = 1
    g = sns.JointGrid(xlim=(0,max), ylim=(0,max))
    sns.scatterplot(x=y_true, y=y_pred, ax=g.ax_joint, s=7, color="xkcd:muted blue")
    sns.histplot(x=y_true, color="xkcd:muted blue", ax=g.ax_marg_x)
    sns.histplot(y=y_pred, color="xkcd:muted blue",  ax=g.ax_marg_y)
    #sns.regplot(x=y_true, y=y_pred, scatter=False, ax=g.ax_joint, line_kws={"color":"xkcd:bluey grey"})
    g.ax_joint.plot([0, max], [0, max], 'k--')
    g.set_axis_labels('True Val', 'Pred Val')
    g.savefig(f"{output_path}/{output_name}.png")

class EarlyStopper:
    def __init__(self, patience:int = 5, min_delta:float = 0.05) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss:float) ->bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



if __name__=="__main__":
    # path_to_fasta = "data/train_test_data/motif_fasta_test_SPLIT_1.fasta"
    # print(one_hot("ACGTN")) 
    # # Should return:
    # # [[1. 0. 0. 0. 0.]
    # # [0. 1. 0. 0. 0.]
    # # [0. 0. 1. 0. 0.]
    # # [0. 0. 0. 1. 0.]]

    # result = create_seq_tensor(path_to_fasta)
    # print(result.shape)
    # print(result)
    out = pd.read_csv("data/outputs/validation_1th_fold_temp_tanh.csv")
    plot_correlation(out.iloc[:,0], out.iloc[:,1], "data/outputs/analysis", "test_correlation")
    #plot_loss_function("data/outputs/logs/training_1th_fold_temp_tanh.log", "data/outputs/analysis", "TEST")