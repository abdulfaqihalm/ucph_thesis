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
from sklearn.metrics import r2_score 
from gensim.models import Word2Vec
from torch.utils import tensorboard
import re

def one_hot(seq: str) -> np.ndarray:
    """
    One-hot encode a sequence

    param: seq: str: sequence to be encoded (ACGTUN)
    return: np.ndarray: one-hot encoded sequence
    """
    look_up_set = set("ACGTN")
    look_up_table = {"A": [1.0, 0.0, 0.0, 0.0],
                     "C": [0.0, 1.0, 0.0, 0.0],
                     "G": [0.0, 0.0, 1.0, 0.0],
                     "T": [0.0, 0.0, 0.0, 1.0],
                     "U": [0.0, 0.0, 0.0, 1.0],
                     "N": [0.0, 0.0, 0.0, 0.0]}
    
    result = []
    for base in seq: 
        if base not in look_up_set:
            print(f"Base {base} is not valid character (ACGTN)")
            result.append([0.0, 0.0, 0.0, 0.0])
            continue
        result.append(look_up_table[base])
    result = np.array(result)
    return result.T

def get_record_by_index(path_to_fasta, target_index):
    seq_records = list(SeqIO.parse(path_to_fasta, format="fasta"))
    if target_index < len(seq_records):
        return str(seq_records[target_index].seq)
    else:
        raise IndexError(f"Invalid index: {target_index}")

def one_hot_to_sequence(one_hot_seq: np.ndarray) -> str:
    """
    Convert one-hot encoding to sequence
    
    param: one_hot_seq: np.ndarray: Expecting the shape of [seq_length, 4]
    return: str: sequence
    """
    if torch.is_tensor(one_hot_seq):
        one_hot_seq = one_hot_seq.numpy()
    if one_hot_seq.shape[1] != 4:
        raise ValueError(f"Invalid shape: {one_hot_seq.shape}. Need to be [seq_length, 4]")
    
    look_up_table = {0: "A", 1: "C", 2: "G", 3: "U"}
    result = []
    for i in range(one_hot_seq.shape[0]):
        base = np.argmax(one_hot_seq[i, :]) # along column of size 4
        result.append(look_up_table[base] if base in look_up_table else "N")
    return "".join(result)

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


def plot_correlation(y_true:np.ndarray, y_pred:np.ndarray, output_path:str="", output_name:str="", title="", interactive=False) -> None:

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
    R2_score = r2_score(y_true, y_pred)
    color_palette = "rocket"
    palette = iter(sns.color_palette(color_palette, 5))
    hist_color = next(palette)

    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(top=0.85)
    g = sns.JointGrid(xlim=(0,max), ylim=(0,max))
    sns.scatterplot(x=y_true, y=y_pred, hue=z, s=6.5, ax=g.ax_joint, palette=color_palette, edgecolor='none', legend=False, alpha=0.3)
    sns.histplot(x=y_true, color=hist_color, edgecolor='none', ax=g.ax_marg_x)
    sns.histplot(y=y_pred, color=hist_color, edgecolor='none', ax=g.ax_marg_y)
    plt.text(0.8*max + 0.07*max, 0.02*max - 0.05, f'r = {corr_coef:.2f}', fontsize=11)
    plt.text(0.8*max + 0.07*max, 0.02*max - 0.1, f'R2 = {R2_score:.2f}', fontsize=11)
    g.set_axis_labels('True Val', 'Pred Val', fontsize=12)
    g.ax_joint.plot([0, max], [0, max], color='black', linestyle='--')
    g.figure.suptitle(title)
    if interactive:
        plt.show()
    else:
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

def create_tensorboard_log_writer(experiment_name: str, 
                  model_name: str, 
                  log_dir: str,
                  extra: str=None) -> tensorboard.writer.SummaryWriter():
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.
    Where timestamp is the current date in YYYY-MM-DD format.

    param: experiment_name (str): Name of experiment.
    param: model_name (str): Name of model.
    param: log_dir (str): Path to save the logs.
    param: extra (str, optional): Anything extra to add to the directory. Defaults to None.

    return: torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join(log_dir, "runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join(log_dir, "runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return tensorboard.writer.SummaryWriter(log_dir=log_dir)

def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    #print("Calculating BMC loss...")
    #breakpoint()
    # print(pred.shape)
    # print(target.shape)
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    # print(logits.is_cuda)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = loss * (2 * noise_var) #.detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

class BMCLoss(torch.nn.Module):
    def __init__(self, init_noise_sigma=1.0):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred.unsqueeze(1), target.unsqueeze(1), noise_var)
    
def calculate_weights(y_train: np.ndarray, bins=np.arange(0, 1, 0.01)) -> np.ndarray:
    """
    Calculate simple weighting from continous-finite data. Code modified from: https://towardsdatascience.com/machine-learning-for-regression-with-imbalanced-data-62629d7ad330

    param: y_train: np.ndarray: target values with shape of [batch_size]
    param: bins: np.ndarray: bins for the target values 
    return: np.ndarray: weights  
    """
    weights = np.zeros(len(y_train))
    idx = np.digitize(y_train, bins=bins)
    for ii in np.unique(idx):
        cond = idx == ii
        weights[cond] = 1 / (np.sum(cond) / len(cond))
    # normalize weights to 1
    weights /= np.sum(weights)
    return weights


import re
import numpy as np 
import torch
from matplotlib import pyplot as plt 
import pandas as pd
import seaborn as sns
def lstm_plot(lstm_param, hidden_size, is_reverse=False, interactive=False):
    is_reverse = "_reverse" if is_reverse else ""
    weight_data = lstm_param[f"kernel_weight{is_reverse}"].T
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [8, 1]}, figsize=(10, 5))
    fig.tight_layout()
    # sns.set(font_scale=1)
    # weights
    ax = sns.heatmap(ax=axs[0], data=weight_data, cmap="PiYG")
    for i in range(0, weight_data.shape[1], hidden_size):
        if i!=0:
            ax.vlines(i, ymin=0, ymax=weight_data.shape[0], colors='black', linewidth=1)
    ax.set_xticks(range(0, weight_data.shape[1],25), range(0, weight_data.shape[1],25))
    ax.set_title(f"Kernel Weight{is_reverse}", weight='bold', size=8)
    ax.set_xlabel("Input  |  Forget  |  Gate  |  Output")
    ax.set_ylabel("Input Units")
    # bias
    bias_data = lstm_param[f"kernel_bias{is_reverse}"].unsqueeze(0)
    ax = sns.heatmap(ax=axs[1], data=bias_data, cmap="PiYG", cbar=False, yticklabels=False, xticklabels=8)
    for i in range(0, bias_data.shape[1], hidden_size):
        if i!=0:
            ax.vlines(i, ymin=0, ymax=bias_data.shape[0], colors='black', linewidth=1)
    ax.set_title(f"Kernel Bias{is_reverse}", weight='bold', size=8)
    ax.set_xlabel("Input  |  Forget  |  Gate  |  Output")
    ax.set_xticks(range(0, bias_data.shape[1],25), range(0, bias_data.shape[1],25))
    plt.subplots_adjust(hspace=0.5)
    if interactive:
        plt.show()
    else:
        return plt

def extract_lstm_info(model):
    lstm = None
    lstm_param  = {}
    is_bidirection = False
    plts = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LSTM):
            lstm = module 
            lstm_layer = lstm.num_layers
            is_bidirection = lstm.bidirectional
            for param in lstm.named_parameters():
                layer = int((re.search(r"l\d+", param[0])).group()[-1])
                h_stat = re.search(r"[i,h]h", param[0]).group()[0]
                if layer == 0:
                    # print(f"Param name: {param[0]} with shape of: {param[1].shape}")
                    if str(param[0]) == "weight_ih_l0":
                        lstm_param["kernel_weight"] = param[1].detach().cpu()
                    if str(param[0]) == "weight_hh_l0":
                        lstm_param["recurrent_weight"] = param[1].detach().cpu()
                    if str(param[0]) == "bias_ih_l0":
                        lstm_param["kernel_bias"] = param[1].detach().cpu()
                    if str(param[0]) == "bias_hh_l0":
                        lstm_param["recurrent_bias"] = param[1].detach().cpu()
                    if str(param[0]) == "weight_ih_l0_reverse":
                        lstm_param["kernel_weight_reverse"] = param[1].detach().cpu()
                    if str(param[0]) == "weight_hh_l0_reverse":
                        lstm_param["recurrent_weight_reverse"] = param[1].detach().cpu()
                    if str(param[0]) == "bias_ih_l0_reverse":
                        lstm_param["kernel_bias_reverse"] = param[1].detach().cpu()
                    if str(param[0]) == "bias_hh_l0_reverse":
                        lstm_param["recurrent_bias_reverse"] = param[1].detach().cpu()
    
    plts["forward_direction"] = lstm_plot(lstm_param,  lstm.hidden_size)
    if is_bidirection:
        plts["backward_direction"] = lstm_plot(lstm_param, lstm.hidden_size, is_reverse=True)
    return plts
                      
                        
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