import numpy as np
import torch
from torch import nn
from wrapper.utils import one_hot_to_sequence


def forward_to_RELU_1(model:torch.nn.Module, x:torch.Tensor) -> torch.Tensor:
    """
    Get the output from the ReLu of the first CNN layer

    param: model: torch.nn.Module: model 
    param: x: torch.Tensor: input tensor

    return: torch.Tensor: output tensor from the ReLu
    """
    for _, layer in enumerate(model.CNN): # Assuming there is defined "self.CNN" layer
        print(layer)
        x = layer(x)
        if isinstance(layer, nn.ReLU): # If it reaches ReLU, return the output 
            return x

def activation_pfm(layer_output: torch.Tensor, one_hot_selected_sequence_idx: torch.Tensor, window: int=9, threshold: float=0.75, by_threshold=False) -> np.ndarray:
    """
    Compute the Position Frequency Matrix (PFM) from a given torch layer output and one-hot-encoded selected_sequence_idx. Details are explained on the DeepBind supplementary material (10.2 Sequence logos). 
    Assuming the output of the CNN layer is "padded same". 
    
    param: layer_output: torch.Tensor, shape: ([batch, seq_length_out, num_cnn_layers])
    param: one_hot_sequence: torch.Tensor, shape: ([batch, seq_length, 4])
    param: window: int, size of the activation window (default 9)
    param: threshold: float, threshold to consider an activation (default 0.5)

    return: PFM: np.array, shape: ([num_filter_layer_output, window, 4])
    """
    input = layer_output.detach().cpu().numpy() # ([batch, seq_length_out, num_cnn_layers])
    X = one_hot_selected_sequence_idx.detach().cpu().numpy() # ([batch, seq_length, 4])
    seq_length = X.shape[1]
    pfm = []
    seq_align_one_hot=[]
    seq_align_text=[]
    total_padding = (window - 1)
    left_pad = total_padding // 2
    # Looping through all kernels -> np(batch,seq_length)
    for filter_index in range(input.shape[2]):
        # Get the maximum score for each selected_sequence_idx
        max_each_seq = np.max(input[:, :, filter_index], axis=1) #[batch]
        if by_threshold:
            max_at_filter = np.max(max_each_seq)
            print(f"Max at filter {filter_index}: {max_at_filter}")
            selected_sequence_idx = np.where(max_each_seq > max_at_filter*threshold)[0] # [batch], in ascending order
            print(f"selected_sequence_idx length: {selected_sequence_idx.shape}")
        else:
            sorted_seq_idx = np.argsort(max_each_seq) #[batch], in ascending order

            # Get the descending order of the sequence index for top 2000 (or less if it's not enough)
            seq_size = sorted_seq_idx.shape[0]
            sort_idx=1
            selected_sequence_idx = []
            while(sort_idx<=2000 and sort_idx<seq_size):
                # Descending
                selected_sequence_idx.append(sorted_seq_idx[-sort_idx])
                sort_idx+=1
            selected_sequence_idx = np.array(selected_sequence_idx)

        temp_seq_align_one_hot = []
        tem_seq_align_text = []
        if len(selected_sequence_idx)>0:
            # print(f"Processing filter: {filter_index}")
            max_indexes = np.argmax(input[list(selected_sequence_idx),:,filter_index], axis = 1) # [batch, seq_lengt] extract max position for each sequence
            # print(f"selected_sequence_idx : {selected_sequence_idx[0:5]}, max_indexes : {max_indexes[0:5]}")
            for seq_index, max_index in zip(selected_sequence_idx, max_indexes):
                # NOTE DIFFERENT 
                start_window = int(max_index) - int(left_pad)
                end_window = start_window + window - 1
                if (end_window > seq_length) or (start_window < 0):
                    # print("brek length, pass")
                    continue
                else : 
                    seq = X[seq_index, start_window:end_window, :] # select the one-hot encoded sequence based on the index
                    temp_seq_align_one_hot.append(seq)
                    tem_seq_align_text.append(one_hot_to_sequence(seq))
                # print(f"one hto seq : {seq}, seq text: {one_hot_to_sequence(seq)}")
                # print(seq)
                # print(one_hot_to_sequence(seq))   
            # print(f"shape of temp align: {np.array(temp_seq_align_one_hot).shape}")
            pfm.append(np.sum(np.array(temp_seq_align_one_hot), axis=0)) # create Position Frequency Matrix, summin over all selected_sequence_idx(batch)

            seq_align_one_hot.append(np.array(temp_seq_align_one_hot))
            seq_align_text.append(np.array(tem_seq_align_text))
        else:
            # If no sequence pass the threshold on a filter, add a zero matrix
            print("No sequence pass the threshold. Adding zero matrix")
            pfm.append(np.zeros((window, 4)))
            seq_align_one_hot.append(np.zeros((window, 4)))
            seq_align_text.append(np.array("N"*window))
    print(np.array(seq_align_text[0]).shape)
    return pfm, seq_align_text, seq_align_one_hot