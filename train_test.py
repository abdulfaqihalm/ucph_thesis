import torch 
import numpy as np
from torch.utils.data import DataLoader
import logging 
from argparse import ArgumentParser, ArgumentTypeError
import os
import csv
from torchmetrics.functional.regression import pearson_corrcoef
from datetime import datetime
from wrapper.utils import EarlyStopper

def test(model: torch.nn.Module, test_loader: DataLoader,
          loss_fn: torch.nn.Module, device: str="cpu") -> tuple[float, dict[str, float]]:
    """
    Test the Model

    param: model: model to be tested
    param: test_loader: test data loader
    param: loss_fn: loss function
    param: device: device location to run the model 

    return: (average loss, metrics: dict[str, float])
    """
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        se = 0.0
        ae = 0.0
        pred = torch.Tensor().to("cpu", non_blocking=True)
        true = torch.Tensor().to("cpu", non_blocking=True)
        for data in test_loader:
            seq = data["seq"].to(device, non_blocking=True)
            meth_case = data["meth_case"].to(device, non_blocking=True)

            meth_pred_response = model(seq)
            loss = loss_fn(meth_pred_response.squeeze(1), meth_case)

            pred = torch.cat([pred, meth_pred_response.squeeze(1).cpu().detach()])
            true = torch.cat([true, meth_case.cpu().detach()])
            
            # Total of average loss from each samples in a batch. Huberloss reduction is mean by default
            test_loss += loss.item()

            for i in range(len(meth_pred_response)):
                y_pred = meth_pred_response[i].cpu().detach().numpy()
                y_true = meth_case[i].cpu().detach().numpy()
                
                # squared and abs error
                se += (y_pred - y_true)**2
                ae += np.abs(y_pred - y_true)

        num_samples = len(test_loader.dataset)
        num_batches = len(test_loader)

        # Average loss per batch in an ecpoh
        test_loss /= num_batches

        # Mean squared error and root mean squared error
        mse =  (se/num_samples).item() 
        rmse = (np.sqrt(se/num_samples)).item()
        
        #logging.info(f"Pred size: {pred.shape}, true size: {true.shape}")
        # Pearson correlation for all of pred and true 
        # x100 for numerical stability
        pearson_corr = pearson_corrcoef(pred, true).item()
        metrics = {"rmse": rmse, 
                   "mse": mse,
                   "pearson_corr": pearson_corr,
                   "pred": pred.numpy(),
                   "true": true.numpy()}
    
    return test_loss, metrics    

def train(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
          loss_fn: torch.nn.Module, save_dir: str, learning_rate: float=1e-3, 
          device: str="cpu", optimizer: torch.nn.Module|None=None, scheduler: torch.optim.lr_scheduler.LRScheduler|None=None, suffix: str=""):
    """
    Train the Model

    param: model: model to be trained
    param: train_loader: train data loader
    param: test_loader: validate data loader
    param: epochs: number of epochs for training 
    param: loss_fn: loss function
    param: save_dir: directory for saving outputs i.e. logs
    param: learning_rate: learning rate for torch.optim
    param: device: device location to run the model 
    param: optimizer: optimizer for updating the parameters

    return: (average loss, metrics: dict[str, float])
    """
    logging.info(f"==== Start training ====")
    
    logfile = open(f"{save_dir}/logs/training_{suffix}.log", "w+")
    logger = csv.DictWriter(logfile, fieldnames=["epoch", "train_loss", "val_loss",  "val_rmse", "val_mse", "val_pearson_corr"])
    logger.writeheader()

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logging.info(f"device {device}")
    if device.type=="cuda":        
        logging.info(f"Set benchmark to True for CUDNN")
        torch.backends.cudnn.benchmark = True
    
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)    
    
    for epoch in range(1, epochs+1):    
        model.train()
        train_loss = 0.0
        # Iterate in batches

        # logging.info(f"Starting batch")
            
        for batch, data in enumerate(train_loader):
            # logging.info(f"Processing batch: {batch}")
            # logging.info(f"Loading data to device: {device}")
            seq = data["seq"].to(device, non_blocking=True)
            meth_case = data["meth_case"].to(device, non_blocking=True)
            # logging.info(f"Finished loading data to device: {device}")
            
            optimizer.zero_grad(set_to_none=True)

            # logging.info(f"Start to predict")
            meth_pred_response = model(seq)
            # logging.info(f"Finish predict")
            loss = loss_fn(meth_pred_response.squeeze(1), meth_case)

            # logging.info(f"Start Backprop")
            loss.backward()
            # logging.info(f"Finish Backprop")
            # logging.info(f"Start Gradient update")
            optimizer.step()
            # logging.info(f"Finish Gradient update")
            # Total of average loss from each samples in a batch. Huberloss reduction is mean by default
            train_loss += loss.item()
        # Average loss per batch in an epoch
        train_loss /= len(train_loader)
        # Run scheduler if provided
        if scheduler:
            scheduler.step()

        validation_loss, metrics = test(model, test_loader, loss_fn, device)

        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.7f}, val_loss: {validation_loss:.7f}, val_mse:{metrics['mse']:.7f}, val_rmse: {metrics['rmse']:.7f}, val_pearson_corr: {metrics['pearson_corr']:.7f}")
        logger.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss, "val_mse": metrics["mse"], 
                         "val_rmse": metrics["rmse"], "val_pearson_corr": metrics["pearson_corr"]})
        
        if early_stopper.early_stop(validation_loss):        
            logging.info(f"Early stop at epoch: {epoch}")     
            break
    
    logfile.close()

    pred_true = {"pred": metrics["pred"], "true": metrics["true"]}

    logging.info(f"==== End training =====")
    return model, pred_true

if __name__=="__main__":
    """
    EXAMPLE USAGE:
    -------
    python train_test.py --data_folder data/train_test_data --num_epochs 500 --batch_size 256 --save_dir data/outputs --mode train --learning_rate 0.001 --gpu 5 --loader_worker 16 --plot y --suffix naive_v1    
    """

    from wrapper.data_setup import SequenceDataset
    from wrapper.utils import plot_loss_function, plot_correlation
    from model import NaiveModelV1, NaiveModelV2, NaiveModelV3, MultiRMModel
    import sys

    # Set logging template
    logging.basicConfig(format='%(asctime)s::%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', 
                        level = logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    # Function to check if file exists from the argparse
    def _file_exists(file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise ArgumentTypeError(f"[EXCEPTION]: The file {file_path} does not exist".format(file_path=file_path))
        return file_path
    
    parser = ArgumentParser(
        description="Running Training and Testing"
    )

    # Function for parsing boolean from argument
    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')
        
    parser.add_argument("-F", "--data_folder", help="Train test Data Folder", required=True)
    parser.add_argument("-E","--num_epochs", help="Max number of epochs", required=True)
    parser.add_argument("-B","--batch_size", help="Number of batch size", default=100, type=int)
    parser.add_argument("-S","--save_dir", help="Directory for saving outputs", required=True)
    parser.add_argument("-M","--mode", help="Training or testing mode",required=True, 
                        choices=["train","test"])
    parser.add_argument("-L","--learning_rate", help="Learning rate", default=0.001, type=float)
    parser.add_argument('-G', "--gpu", default=[0,1], nargs='+', type=int)
    parser.add_argument('-W', "--loader_worker", default=0, type=int)
    parser.add_argument('-I', "--m6A_info", default='no',const='no', nargs='?', 
                        choices=['flag_channel', 'level_channel', 'add_middle', 'no'], 
                        help="Include m6A info? ['flag_channel', 'level_channel', 'add_middle', 'no']")
    #parser.add_argument('-I', "--m6A_info", type=str2bool, nargs='?', const=True, default=False, help="Include m6A info? [y/n]")
    parser.add_argument('-P', "--plot", type=str2bool, nargs='?', const=True, default=False, help="Plot loss or not [y/n]")
    parser.add_argument('-SF', "--suffix", default="", help="Suffix for output files")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)    
    data_folder = args.data_folder
    m6A_info = args.m6A_info
    num_epochs = int(args.num_epochs)
    loader_worker = int(args.loader_worker)
    batch_size = args.batch_size
    save_dir = args.save_dir
    suffix = args.suffix
    mode = args.mode
    plot = args.plot
    learning_rate = args.learning_rate


    device = (torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu"))
    logging.info(f"Running on device: {device}")  

    if mode=="train":
        logging.info(f"Training mode")
        for i in range(1, 2): #5-folds SHOULD BE 6
            logging.info(f"Fold-{i}")
            seq_fasta_train_path = f"{data_folder}/motif_fasta_train_SPLIT_{i}.fasta"
            promoter_fasta_train_path = f"{data_folder}/promoter_fasta_train_SPLIT_{i}.fasta"
            # label_train_json_path = f"{data_folder}/train_label_SPLIT_{i}.json"
            label_train_json_path = f"{data_folder}/train_meta_data_SPLIT_{i}.json"
            m6A_info_train_path = None

            seq_fasta_test_path = f"{data_folder}/motif_fasta_test_SPLIT_{i}.fasta"
            promoter_fasta_test_path = f"{data_folder}/promoter_fasta_test_SPLIT_{i}.fasta"
            # label_test_json_path = f"{data_folder}/test_label_SPLIT_{i}.json"
            label_test_json_path = f"{data_folder}/test_meta_data_SPLIT_{i}.json"
            m6A_info_test_path = None
            
            # ['flag_channel', 'level_channel', 'add_middle', 'no']
            if m6A_info=="level_channel":
                m6A_info_train_path = f"{data_folder}/train_case_m6A_prob_data_SPLIT_{i}.json"
                m6A_info_test_path  = f"{data_folder}/test_case_m6A_prob_data_SPLIT_{i}.json"
            if m6A_info=="flag_channel":
                m6A_info_train_path = f"{data_folder}/train_case_m6A_flag_data_SPLIT_{i}.json"
                m6A_info_test_path  = f"{data_folder}/test_case_m6A_flag_data_SPLIT_{i}.json"

            train_dataset = SequenceDataset(seq_fasta_train_path, promoter_fasta_train_path, label_train_json_path, m6A_info_train_path, m6A_info)
            test_dataset = SequenceDataset(seq_fasta_test_path, promoter_fasta_test_path, label_test_json_path, m6A_info_test_path, m6A_info)
       
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_worker)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_worker)
            logging.info(f"Number of training batches: {len(train_loader)}, Total training sample: {len(train_loader.dataset)}")
            logging.info(f"Number of test batches: {len(test_loader)}, Total test sample: {len(test_loader.dataset)}")

            # 0.5MSE if delta<0.1, otherwise delta(|y-y_hat| - 0.5delta)
            loss_fn = torch.nn.HuberLoss(delta=1)
            # [HARD CODED] Input size here is hard coded for the naive model based on the data
            # 1001 means we have 500 down-up-stream nts
            #model = NaiveModelV2() 
            #model = MultiRMModel(num_task=1)


            if m6A_info: 
                logging.info(f"Using m6A info")
                model = NaiveModelV2(input_channel=5, cnn_first_filter=8)
            else:
                model = NaiveModelV2(input_channel=4, cnn_first_filter=8)

            model.to(device)
            #model=torch.nn.DataParallel(model) 




            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # here added the weight dacay
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # here added the weight dacay
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5) # here added the weight dacay for temp_w_l2reg. for temp no.
            #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.1) # here added exponenetial dcay for the optimizer
            # Train and Validate the model
            _, pred_true = train(model, train_loader, test_loader, num_epochs, loss_fn, save_dir, learning_rate, device, optimizer, suffix=f"{i}th_fold_{suffix}")

            model_file = f"{save_dir}/models/trained_model_{i}th_fold_{suffix}.pkl"
            torch.save(model.state_dict(), model_file)
            logging.info(f"Model of {i}th fold saved to {model_file}")

            if plot:
                plot_loss_function(f"{save_dir}/logs/training_{i}th_fold_{suffix}.log", f"{save_dir}/analysis", f"loss_plot_{i}th_fold_{suffix}")
                plot_correlation(pred_true["true"], pred_true["pred"], f"{save_dir}/analysis", f"correlation_plot_{i}th_fold_{suffix}")

            save_val = True
            if save_val:
                with open(f"{save_dir}/validation_{i}th_fold_{suffix}.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_true["true"], pred_true["pred"]))
            logging.info(f"Finished on Fold-{i}")
    else:
        # TODO: Implement testing mode using the pickled model
        logging.info(f"Loading test data ")