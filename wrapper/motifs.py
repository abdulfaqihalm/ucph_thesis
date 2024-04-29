import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import os


def activation_pfm(layer_output: torch.Tensor, one_hot_sequences: torch.Tensor, window: int = 9, threshold: float = 0.5) -> np.ndarray:
    """
    Compute the Position Frequency Matrix (PFM) from a given torch layer output and one-hot-encoded sequences. Details are explained on the DeepBind supplementary material (10.2 Sequence logos). 

    input: layer_output: torch.Tensor, shape: ([batch, seq_length_out, num_cnn_layers])
    input: one_hot_sequence: torch.Tensor, shape: ([batch, seq_length, 4])
    input: window: int, size of the activation window (default 9)
    input: threshold: float, threshold to consider an activation (default 0.5)

    return: PFM: np.array, shape: ([num_filter_layer_output, window, 4])
    """
    input = layer_output  # ([batch, seq_length_out, num_cnn_layers])
    X = one_hot_sequences  # ([batch, seq_length, 4])

    seq_length = X.shape[1]
    pfm = []
    window_left = int(window/2)
    window_right = window - window_left
    # Looping through all kernels -> np(batch,seq_length)
    for filter_index in range(input.shape[2]):
        # extract coordinates (sequence, position) which pass threshold
        x, y = np.where(input[:, :, filter_index] > threshold)
        sequences = set(x)  # extract sequence which pass threshold
        if len(sequences) > 0:
            # extract max position for each sequence
            max_indexes = np.argmax(
                input[list(sequences), :, filter_index], axis=1)
            seq_align = []
            for seq_index, max_index in zip(sequences, max_indexes):
                start_window = int(max_index) - window_left
                end_window = int(max_index) + window_right

                # Create padding if out of bound
                if start_window < 0:
                    padding = np.zeros((-start_window, 4))
                    seq = np.concatenate(
                        [padding, X[seq_index, :end_window, :]], axis=0)
                elif end_window > seq_length:
                    padding = np.zeros((end_window - seq_length, 4))
                    seq = np.concatenate(
                        [X[seq_index, start_window:, :], padding], axis=0)
                else:
                    seq = X[seq_index, start_window:end_window, :]
                seq_align.append(seq)

            if len(seq_align) > 1:
                # create Position Frequency Matrix, summin over all sequences(batch)
                pfm.append(np.sum(seq_align, axis=0))
            else:
                pfm.append(seq_align)
        else:
            # If no sequence pass the threshold on a filter, add a zero matrix
            print("No sequence pass the threshold. Adding zero matrix")
            pfm.append(np.zeros((window, 4)))

    return np.array(pfm)


def plot_motifs_from_pfm(pfm: np.ndarray, out_dir: str, file_name: str, interactive: bool = True) -> None:
    """
    Plot motifs from Position Frequency Matrix.
    modified from https://github.com/p-koo/learning_sequence_motifs/blob/master/code/deepomics/visualize.py

    input: pfm: np.ndarray, shape: ([num_filter_layer_output, window, 4])
    input: out_dir: str, output directory
    input: file_name: str, output file name
    """
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle("Learned Motifs from The First Layer of CNN", fontsize=20)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    num_filters = pfm.shape[0]
    num_cols = 5
    num_rows = int(np.ceil(num_filters/num_cols))
    for n, f in enumerate(pfm):
        # f in [window.4] shape
        ax = fig.add_subplot(num_rows, num_cols, n+1)
        # IC is the information content
        # calculated from relative entropy: \sum{bases}p(a)log_{2}\frac{p(a), background_freq(a)}
        # which is the same as \sum{bases}p(a)log_{2}{p(a)} + log_{2}{4}, assuming a uniform background distribution
        n_bases = 4
        pfm_sum = np.sum(f, axis=1, keepdims=True)
        ppm = f / pfm_sum  # calculating PPM
        IC = np.log2(n_bases) + np.sum(ppm * np.log2(ppm+1e-8),
                                       axis=1, keepdims=True)  # avoid log(0) by add small value
        logo = IC*ppm

        counts_df = pd.DataFrame(
            data=logo, columns=list("ACGT"), index=list(range(L)))

        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("none")
        ax.xaxis.set_ticks_position("none")
        plt.title(f"FILTER {n+1}")
        plt.xticks([])
        plt.yticks([])

    if interactive:
        plt.show()
    else:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        outfile = os.path.join(out_dir, f"{file_name}.pdf")
        fig.savefig(outfile, format="pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)