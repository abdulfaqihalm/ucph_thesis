import torch 
import numpy as np
import logging 
from argparse import ArgumentParser, ArgumentTypeError
import os
import csv
from wrapper import utils
from torchmetrics.functional.regression import pearson_corrcoef
from train_test import test
from wrapper.utils import plot_loss_function, plot_correlation
import sys
from model import NaiveModelV2
from torch.utils.data import DataLoader
from wrapper.data_setup import SequenceInferenceDataset


if __name__=="__main__":
    """
    Example: python test.py --model_path data/outputs/trained_model_3th_fold_naive_v2_500_wo_m6A_control_2.pkl --data_path data/train_test_data_500_2/motif_fasta_test_SPLIT_3.fasta --gpu 4 --suffix _500_wo_m6A_control_2
    """

    # Set logging template
    logging.basicConfig(format='%(asctime)s::%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', 
                        level = logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    parser = ArgumentParser(
        description="Test model"
    )

    parser.add_argument("-M", "--model_path", help="Path to pickled model and its weight", required=True)
    parser.add_argument("-D", "--data_path", help="Path to the sequence data that will be predicted", required=True)
    parser.add_argument('-G', "--gpu", default=[0,1], nargs='+', type=int)
    parser.add_argument('-SF', "--suffix", default="", help="Suffix for output files")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    model_path = args.model_path
    data_path = args.data_path
    suffix = args.suffix


    device = (torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu"))
    logging.info(f"Running on device: {device}")  

    # Load the model
    model = NaiveModelV2(1001)
    #print(torch.load(model_path))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    pred_list = []
    test_dataset = SequenceInferenceDataset(data_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            #print(pred.squeeze().cpu().numpy())
            pred_list.extend(pred.squeeze().cpu().numpy())


    np.savetxt(f"data/outputs/prediction{suffix}.csv", pred_list, delimiter=",")
