# UCPH Master Thesis repository

## Introduction
This repository contains the code for my master's thesis with a title "Neural Network for Investigating Sequence Determinants of m6A Redistribution During Hypoxia Response". In a nutshell, given a sequence which contains either m6A site of normal or hypoxia condition in HeLa cell from GLORI paper [1], we want to create a model which could predict the stoichiometry for both conditions. Moreover, we also want to know what are the motifs which might drive the difference between the two conditions.


## Instruction to Run

### Libraries
First of all, you need to install the required library for both python and R.
For python you can just run it with conda
```shell
conda env create -f environment_gpu.yaml
```
For R, just install all the packages from two main R files manually.


### Data Preparation
To prepare the data, you need to run the train_test_split.R that processes the GLORI raw data and split it into 5-folds based on the gene: 
```shell
Rscript train_test_split.R
```
It will output splitted data of the sequence in .FASTA format for train and validation process (in `motif_fasta_<train/test>_SPLIT_<fold>.fasta`), their promotors (in `promoter_fasta_<train/test>_SPLIT_<fold>.fasta`). In addition, it will aso give you the metadata in json format ((in `<train/test>_meta_data_SPLIT_<fold>.json`))


#### Model Training and Tuning
Next, if you want to reproduce the hyperparameter optimization problem, you can run:
```shell
python tune_dual.py --data_folder data/dual_outputs --embedding one-hot --tune_output_path ray_results --tune_output_folder_name tuning --num_samples 50 --max_num_epochs 16
```

You change the num_samples param and config variable inside the code. You can find the summary of the tuning process in ray_results in `ray_results/tuning/results_summary.csv`. Since I haven't created the feature to save the tuned parameters into a file (which using Ray's checkpoint) you need to rerun it via other python script (which also allows you to experiment with your provided config). To do that just run:
```shell
python train_test_dual.py --data_folder data/dual_outputs --num_epochs 200 --batch_size 128 --save_dir data/outputs --mode train --learning_rate 0.001 --gpu 2 --plot y --loader_worker 6 --kaiming_init y --tensorboard_writer y  --suffix <your_suffix>
```


The above script will give you the two logs: the csv logs (under outputs/logs) and the tensorlogs (which located under outputs/tensorboard_logs/runs/<date>). The tensorlog is enabled if you type `--tensorboard_writer y `. If you have the tensor log, you can see the logs for your runs (and also all the folds) by running:  
```shell
tensorboard --logdir=data/outputs/tensorboard_logs/runs --load_fast=false
 
```  

However, if you are using ssh, you can bind the addess with you local computer
```shell

ssh -L serverport:localhost:localport
```
where serverport should be equal to the exposed_port on tensorboard command. Then you can see the board at `localhost:localport` on your local device. It will give you the loss, pearson corr, and also LSTM interpretation from the trained model.


Furthermore, you can also see additional outputs from the last python script inside the outputs folder. If you type `--plot y` it will produce the validation loss and also correlation between the true and predicted value of normal and hypoxia conditions. The plots' path are located at `data/outputs/analysis/correlation_plot<fold>_suffix.png` and `data/outputs/analysis/loss_plot_<fold>_suffix.png`. Aside that, the last python script also outputs the pickled trained model parameter inside `data/outputs/model/<model>` as well as the prediction csv (`data/outputs/predictions/<file>`) file for further do


### Model Interpretation
For the model interpretation, you can use the `analysis/model_interpretation.ipynb` to output the ISM analyses which contain the scatter plot of m6A level difference, logos of attribution score, and the heatmap of the m6A level difference average for each position.


### Motif Importances
To know further what are the drivers of the dynamic of m6A levels, I created a linear model assuming linear interaction between the motifs and the m6A level difference. The motifs are extracted from the first layer of CNN. To do that you cnat just run the code inside `analysis/motifs_importance.ipynb`


### Result Analysis
All the post-hoc analysis can be found on `result_analysis.Rmd`

## References 
- Cong Liu, Hanxiao Sun, Yunpeng Yi, et al. „Absolute quantification of single-base m6A methylation in the mammalian transcriptome using GLORI“. In: Nature Biotechnology 41.3 (Mar. 2023), pp. 355–366.