from Bio import SeqIO
import numpy as np 
import torch
from matplotlib import pyplot as plt 
import pandas as pd
import seaborn as sns
import random
import os
from scipy.stats import gaussian_kde
from .gene2vec_embedding import gene2vec
from gensim.models import Word2Vec

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

def get_record_by_index(path_to_fasta, target_index):
    seq_records = list(SeqIO.parse(path_to_fasta, format="fasta"))
    if target_index < len(seq_records):
        return str(seq_records[target_index].seq)
    else:
        raise IndexError(f"Invalid index: {target_index}")

def create_seq_tensor(path_to_fasta: str, idx: int|None=None, transform: str="one-hot", path_to_embedding: str|None=None) -> torch.Tensor:
    """
    Create a tensor of sequences from a fasta file

    param:  path_to_fasta: str: path to the fasta file
    param:  idx: int: index (row) of the sequence on the fasta file
    param:  transform: str: transformation method ("one-hot" or "gene2vec")
    param:  path_to_embedding: str: path to the embedding file (required if transform is "gene2vec")
    return: torch.Tensor: tensor of sequences
    """
    if transform == "one-hot":
        result = []
        if idx is None:
            seq_records = list(SeqIO.parse(path_to_fasta, format="fasta"))
            for seq_record in seq_records:
                try:
                    result.append(one_hot(str(seq_record.seq)))
                except ValueError as e:
                    print(e)
        else:
            try:
                result = one_hot(get_record_by_index(path_to_fasta, idx))
            except IndexError:
                raise ValueError(f"Invalid index: {idx}")
    elif transform == "gene2vec":
        if path_to_embedding is None:
            raise ValueError("Path to embedding file is required for gene2vec transformation")
        result = []
        # print("Loading word2vec model")
        embedding = Word2Vec.load(path_to_embedding)
        # print("Finished loading word2vec model")
        if idx is None:
            seq_records = list(SeqIO.parse(path_to_fasta, format="fasta"))
            for seq_record in seq_records:
                try:
                    result.append(gene2vec(str(seq_record.seq), embedding))
                except ValueError as e:
                    print(e)
        else:
            try:
                result = gene2vec(get_record_by_index(path_to_fasta, idx), embedding)
            except IndexError:
                raise ValueError(f"Invalid index: {idx}")
    else:
        raise ValueError(f"Invalid transform option: {transform}")
    
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


def plot_correlation(y_true:np.ndarray, y_pred:np.ndarray, output_path:str, output_name:str, title="") -> None:
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
    if np.max(y_true) > 1 and np.max(y_pred) > 1:
        max = 100
    else:
        max = 1

    # Calculate the point density
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    color_palette = "rocket"
    palette = iter(sns.color_palette(color_palette, 5))
    hist_color = next(palette)

    plt.figure()

    g = sns.JointGrid(xlim=(0,max), ylim=(0,max))
    sns.scatterplot(x=y_true, y=y_pred, hue=z, s=6.5, ax=g.ax_joint, palette=color_palette, edgecolor='none', legend=False, alpha=0.3)
    sns.histplot(x=y_true, color=hist_color, edgecolor='none', ax=g.ax_marg_x)
    sns.histplot(y=y_pred, color=hist_color, edgecolor='none', ax=g.ax_marg_y)
    plt.text(80, 5, f'r = {corr_coef:.2f}', fontsize=12)
    g.set_axis_labels('True Val', 'Pred Val', fontsize=12)
    g.ax_joint.plot([0, max], [0, max], color='black', linestyle='--')
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
    
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__=="__main__":
    from time import time
    path_to_fasta = "/binf-isilon/renniegrp/vpx267/ucph_thesis/data/dual_outputs/motif_fasta_train_SPLIT_1.fasta"
    # print(one_hot("ACGTN")) 
    # t = time()
    # print(create_seq_tensor(path_to_fasta).shape)
    # # torch.Size([103855, 4, 1001])
    # # Time load data: 0.81 mins
    # print('Time load data: {} mins'.format(round((time() - t) / 60, 2))) 
    # Should return:
    # [[1. 0. 0. 0. 0.]
    # [0. 1. 0. 0. 0.]
    # [0. 0. 1. 0. 0.]
    # [0. 0. 0. 1. 0.]]
    # path_to_embedding = "/binf-isilon/renniegrp/vpx267/ucph_thesis/data/embeddings/gene2vec/double_outputs/split_1.model"
    # gene2vec_result = gene2vec("ACGTN", Word2Vec.load(path_to_embedding)) # Expect to return 3x300 
    # print(gene2vec_result.shape)
    # # # print(gene2vec_result)


    # t = time()
    # result = create_seq_tensor(path_to_fasta, transform="gene2vec", path_to_embedding=path_to_embedding)
    # print(result.shape)
    # # torch.Size([103855, 300, 999])
    # # Time load data: 10.19 mins
    # print('Time load data: {} mins'.format(round((time() - t) / 60, 2))) 
    # print(result[0])

    # result = create_seq_tensor(path_to_fasta)
    # print(result.shape)
    # print(result)
    # out = pd.read_csv("/binf-isilon/renniegrp/vpx267/ucph_thesis/data/outputs/validation_1th_fold_case_m6_info-no_promoter-False_single_model_TEST_BEST_PARAM.csv")
    # plot_correlation(out.iloc[:,0], out.iloc[:,1], "data/outputs/analysis", "test_correlation")
    #plot_loss_function("data/outputs/logs/training_1th_fold_temp_tanh.log", "data/outputs/analysis", "TEST")




    df = pd.read_csv("/binf-isilon/renniegrp/vpx267/ucph_thesis/data/outputs/predictions/validation_1th_fold_dual_outputs_m6_info-no_promoter-False_ONE_HOT_TEST_DUAL_OUTPUTS_BEST_PARAMS_NEW.csv")
    df.head()
    print(df.iloc[:,0].values)
    plot_correlation(df.iloc[:,0], df.iloc[:,2], "data/outputs/analysis", "test_correlation_dual_outputs_control", "control")
    plot_correlation(df.iloc[:,1], df.iloc[:,3], "data/outputs/analysis", "test_correlation_dual_outputs_case", "case")