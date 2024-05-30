from tangermeme.ism import saturation_mutagenesis
import logomaker
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def attribution_score(y0: torch.Tensor, y_hat:torch.Tensor, target:int) -> torch.Tensor:
	"""ISM attributions calculation
	
	param: y0: torch.Tensor: reference output
	param: y_hat: torch.Tensor: ISM generated output
	param: target: int: target class (0 normal, 1 hypoxia)

	return: attr: torch.Tensor: attribution score
	"""

	attr = y_hat[:, :, :, target] - y0[:, None, None, target]
	attr -= torch.mean(attr, dim=1, keepdims=True)

	if len(attr.shape) > 3:
		attr = torch.mean(attr, dim=tuple(range(3, len(attr.shape))))
	return attr

def plot_ism_scatter(y_delta:torch.Tensor, target:int, fold:int, slice_start:int = 250, slice_end:int = 751) -> None:
	"""Plot ISM Scatter plot
	
	param: y_delta: torch.Tensor: difference between ISM generated output and reference output
	param: target: int: target class (0 normal, 1 hypoxia)
	param: fold: int: fold number
	param: slice_start: int: start position of slice
	param: slice_end: int: end position of slice
	"""
	# Example data (replace this with your actual numpy array)
	group = "hypoxia" if target==1 else "normal"
	data = y_delta[:,:,:,target]
	data = data.reshape(-1,1001)
	data = data[:,slice_start:slice_end].numpy()
	x = list(range(slice_start-500,slice_end-500))
	# Plotting each row against xfig, ax = plt.subplots(figsize=(45, 5))
	fig, ax = plt.subplots(figsize=(24, 10))
	for i in range(data.shape[0]):
		# Determine colors for each point based on its sign
		colors = ['green' if value > 0 else 'red' for value in data[i]]
		
		# Scatter plot with conditional coloring
		ax.scatter(x, data[i], color=colors, alpha=0.3, s=3)

	ax.set_facecolor("white")
	# Adding labels and legend
	ax.set_xlabel('Nucleotide Position Centered at m6A Site')
	ax.set_ylabel('Predicted m6A Level Change')
	ax.legend()

	plt.rcParams.update({'font.size': 12})
	plt.savefig(f"ism/{group}_delta_m6A_fold_{fold}_{slice_start}_{slice_end}.png",bbox_inches='tight')
	# Show the plot
	plt.show()

def plot_score_heatmap(attr:torch.Tensor, target:int, fold:int, slice_start:int = 450, slice_end:int = 551) -> None:
	"""
	Plot the attribution score heatmap
	
	param: attr: torch.Tensor: attribution score 
	param: target: int: target class (0 normal, 1 hypoxia)
	param: fold: int: fold number
	param: slice_start: int: start position of slice
	param: slice_end: int: end position of slice
	"""
	group = "hypoxia" if target==1 else "normal"
	slice_start = slice_start
	slice_end = slice_end
	plot_df = pd.DataFrame(torch.mean(attr[:,:,:,target], dim=0).numpy().T, columns=list("ACGU"))

	fig, ax = plt.subplots(figsize=(45, 5))

	sns.set(font_scale=1.5, style='white')
	s = sns.heatmap(data=(plot_df.T).iloc[:,slice_start:slice_end], xticklabels=list(range(slice_start-500,slice_end-500)), cmap="RdBu_r", cbar_kws={"pad": 0.01, "label":"Average Predicted\nm6A Level Change"})
	s.set(xlabel='Nucleotide Position Centered at m6A Site')
	plt.savefig(f"ism/{group}_avg_score_ism_fold_{fold}.png",bbox_inches='tight')
	plt.show()

def plot_logos(y_ref:torch.Tensor, y_ism:torch.tensor, input_sequences:torch.Tensor, target:int, fold:int, slice_start:int = 450, slice_end:int = 551) -> None:
	"""
	Plot the logos of attribution score
	
	param: y_ref: torch.Tensor: data generated from the reference 
	param: y_ism: torch.Tensor: data generated from the ISM
	param: fold: int: fold number
	param: slice_start: int: start position of slice
	param: slice_end: int: end position of slice
	"""
	group = "hypoxia" if target==1 else "normal"
	slice_start = slice_start
	slice_end = slice_end
	attr = attribution_score(y_ref, y_ism, target)
	# This is important to only show nucleotide which only appear on the input sequences i.e. m6A site only has "A"
	attr = attr * input_sequences
	plot_logo = np.average(attr, axis=0)
	plot_logo_df = pd.DataFrame(plot_logo.T, columns=list("ACGU"))
	fig, ax = plt.subplots(figsize=(55, 5))
	logomaker.Logo((plot_logo_df).iloc[slice_start:slice_end], ax=ax, color_scheme='classic')
	ax.xaxis.set_ticks_position("none")
	ax.set_facecolor("white")
	ax.set_ylabel('Attribution Score')
	plt.xticks(rotation=90)
	plt.rcParams.update({'font.size': 16})
	plt.grid(False)
	plt.xticks(list(range(slice_start, slice_end)), list(range(slice_start-500, slice_end-500)))  
	plt.savefig(f"ism/{group}_logo_ism_fold_{fold}.png" ,bbox_inches='tight')
	plt.show()

def ism_analysis(model:torch.nn.Module, input_sequences:torch.Tensor, fold:int) -> None: 
    """
    Run ISM analysis given input sequences in one-hot encoded format

	param: model: torch.nn.Module: trained model 
	param: input_sequences: torch.Tensor: one-hot encoded input sequences 
	param: fold: int: fold number
    """
    y_ref, y_ism = saturation_mutagenesis(model, input_sequences, raw_outputs=True)
    y_delta = y_ism - y_ref[:, None, None]
    print("Saving...")
    np.savez(f"ism/ism_raw_fold_{fold}_fixed_tune.npz", y_delta=y_delta, y_ism=y_ism, y_ref=y_ref)
    print("Saving Done")
	
    plot_score_heatmap(y_delta, 0, fold, slice_start = 450, slice_end = 551)
    plot_score_heatmap(y_delta, 1, fold, slice_start = 450, slice_end = 551)

    plot_logos(y_ref, y_ism, input_sequences, 0, fold, slice_start = 450, slice_end = 551)
    plot_logos(y_ref, y_ism, input_sequences, 1, fold, slice_start = 450, slice_end = 551)

    plot_score_heatmap(y_delta, 0, fold, slice_start = 250, slice_end = 751)
    plot_score_heatmap(y_delta, 1, fold, slice_start = 250, slice_end = 751)
    plot_score_heatmap(y_delta, 0, fold, slice_start = 450, slice_end = 551)
    plot_score_heatmap(y_delta, 1, fold, slice_start = 450, slice_end = 551)



    plot_ism_scatter(torch.Tensor(y_delta), 0, fold, slice_start = 450, slice_end = 551)
    plot_ism_scatter(torch.Tensor(y_delta), 1, fold, slice_start = 450, slice_end = 551)

    plot_ism_scatter(torch.Tensor(y_delta), 0, fold, slice_start = 250, slice_end = 751)
    plot_ism_scatter(torch.Tensor(y_delta), 1, fold, slice_start = 250, slice_end = 751)
    