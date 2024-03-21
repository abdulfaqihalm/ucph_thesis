import torch 
import numpy as np
from torch.utils.data import DataLoader
import logging 
from argparse import ArgumentParser, ArgumentTypeError
import os
import csv
from torchmetrics.wrappers import MultioutputWrapper
from torchmetrics.regression import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError
from datetime import datetime
from wrapper.utils import EarlyStopper

def test(model: torch.nn.Module, test_loader: DataLoader,
          loss_fn: torch.nn.Module, target: None|str = "case", device: str="cpu") -> tuple[float, dict[str, float]]:
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
        pred= torch.Tensor().to("cpu", non_blocking=True)
        true = torch.Tensor().to("cpu", non_blocking=True)

        for data in test_loader:
            seq = data["seq"].to(device, non_blocking=True)
            meth_true_val = torch.stack([data[f"meth_control"].to(device, non_blocking=True), 
                                            data[f"meth_case"].to(device, non_blocking=True)], dim=1)
            meth_pred_val = model(seq)
            loss1 = loss_fn(meth_pred_val[:,0], meth_true_val[:,0])
            loss2 = loss_fn(meth_pred_val[:,1], meth_true_val[:,1])
            loss = loss1 + loss2

            pred = torch.cat([pred, meth_true_val.cpu().detach()])
            true = torch.cat([true, meth_pred_val.cpu().detach()])
            
            # Total of average loss from each samples in a batch. Huberloss reduction is mean by default
            test_loss += loss.item()

        # print(pred)
        # print(true)
        # print(pred.shape)
        # print(true.shape)

        # Average loss per batch in an epoch
        test_loss /= len(test_loader)

        pearson_corr = MultioutputWrapper(PearsonCorrCoef(), 2)
        mse = MultioutputWrapper(MeanSquaredError(), 2)
        mae = MultioutputWrapper(MeanAbsoluteError(), 2)
        
        pearson_corr = pearson_corr(pred, true)
        mse = mse(pred, true)
        mae = mae(pred, true)
        rmse = torch.sqrt(mse)
        metrics = {"rmse": rmse.numpy(), 
                   "mse": mse.numpy(),
                   "mae": mae.numpy(),
                   "pearson_corr": pearson_corr.numpy(),
                   "pred": pred.numpy(),
                   "true": true.numpy()}
    
    return test_loss, metrics    

def train(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
          loss_fn: torch.nn.Module, save_dir: str, target: None|str = "case", learning_rate: float=1e-3, 
          device: str="cpu", optimizer: torch.nn.Module|None=None, scheduler: torch.optim.lr_scheduler.LRScheduler|None=None, suffix: str="") -> tuple[torch.nn.Module, dict[str, float]]:
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

    return: tuple[torch.nn.Module, dict[str, float]]
    """
    logging.info(f"==== Start training ====")
    logfile = open(f"{save_dir}/logs/training_{suffix}.log", "w+")
    logger = csv.DictWriter(logfile, fieldnames=["epoch", "train_loss", "val_control_loss", "val_control_rmse", "val_control_mse", "val_control_mae", "val_control_pearson_corr", "val_case_loss", "val_case_rmse", "val_case_mse", "val_case_mae", "val_case_pearson_corr"])
    logger.writeheader()

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    early_stopper = EarlyStopper(patience=3, min_delta=0.01)    
    
    for epoch in range(1, epochs+1):    
        model.train()
        train_loss = 0.0
            
        for batch, data in enumerate(train_loader):
            seq = data["seq"].to(device, non_blocking=True)
            meth_true_val = torch.stack([data[f"meth_control"].to(device, non_blocking=True), 
                                            data[f"meth_case"].to(device, non_blocking=True)], dim=1)
            
            optimizer.zero_grad(set_to_none=True)
            meth_pred_val = model(seq)
            # print(meth_pred_val.shape)
            # print(meth_pred_val)
            # print(meth_true_val.shape)
            # print(meth_true_val)
            loss1 = loss_fn(meth_pred_val[:,0], meth_true_val[:,0])
            loss2 = loss_fn(meth_pred_val[:,1], meth_true_val[:,1])
            loss = loss1 + loss2
            # logging.info(f"Finished loading data to device: {device}")
            # print(meth_pred_val) 
            # print(meth_true_val)
            # logging.info(f"Finish predict")
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

        validation_loss, metrics = test(model=model, test_loader=test_loader, loss_fn=loss_fn, 
                                        target=target, device=device)

        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.7f}, \n val_loss: {validation_loss:.7f} \n val_control_mse:{metrics['mse'][0].item():.7f}, val_control_rmse: {metrics['rmse'][0].item():.7f}, val_control_mae: {metrics['mae'][0].item():.7f}, val_control_pearson_corr: {metrics['pearson_corr'][0].item():.7f} \n val_case_mse:{metrics['mse'][1].item():.7f}, val_case_rmse: {metrics['rmse'][1].item():.7f}, val_case_mae: {metrics['mae'][1].item():.7f}, val_case_pearson_corr: {metrics['pearson_corr'][1].item():.7f}")
        logger.writerow({"epoch": epoch, "train_loss": train_loss, "val_control_mse":metrics['mse'][0], "val_control_rmse": metrics['rmse'][0], "val_control_mae": metrics['mae'][0], "val_control_pearson_corr": metrics['pearson_corr'][0],"val_case_mse":metrics['mse'][1], "val_case_rmse": metrics['rmse'][1], "val_case_mae": metrics['mae'][1], "val_case_pearson_corr": metrics['pearson_corr'][1]})
        
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

    without m6A channel
    -------
    python train_test.py --data_folder data/train_test_data_500_2 --num_epochs 200 --batch_size 256 --save_dir data/outputs --mode train --learning_rate 0.0001 --gpu 5 --loader_worker 4 --plot y --target case --suffix temp_test_code_no_m6A
    

    with m6A channel
    -------
    python train_test.py --data_folder data/train_test_data_500_2 --num_epochs 3 --batch_size 256 --save_dir data/outputs --mode train --learning_rate 0.0001 --gpu 5 --loader_worker 4 --plot y --m6A_info level_channel -- --suffix temp_test_code
    """

    from wrapper.data_setup import SequenceDatasetDual
    from wrapper.utils import plot_loss_function, plot_correlation, seed_everything
    from model import NaiveModelV1, NaiveModelV2, NaiveModelV3, MultiRMModel, ConvTransformerModel, ConfigurableModel
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
        
    parser.add_argument("--data_folder", help="Train test Data Folder", required=True)
    parser.add_argument("--num_epochs", help="Max number of epochs", required=True)
    parser.add_argument("--batch_size", help="Number of batch size", default=100, type=int)
    parser.add_argument("--save_dir", help="Directory for saving outputs", required=True)
    parser.add_argument("--mode", help="Training or testing mode",required=True, 
                        choices=["train","test"])
    parser.add_argument("--learning_rate", help="Learning rate", default=0.001, type=float)
    parser.add_argument("--gpu", default=[0,1], nargs='+', type=int)
    parser.add_argument("--loader_worker", default=0, type=int)
    parser.add_argument("--add_promoter", type=str2bool, nargs='?', const=True, default=False, help="Append promoter sequence[y/n]")
    parser.add_argument("--m6A_info", default='no',const='no', nargs='?', 
                        choices=['flag_channel', 'level_channel', 'add_prob_middle', 'add_flag_middle', 'no'], 
                        help="Include m6A info? ['flag_channel', 'level_channel', 'add_middle', 'no']")
    #parser.add_argument("--m6A_info", type=str2bool, nargs='?', const=True, default=False, help="Include m6A info? [y/n]")
    parser.add_argument("--plot", type=str2bool, nargs='?', const=True, default=False, help="Plot loss or not [y/n]")
    parser.add_argument("--suffix", default="", help="Suffix for output files")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)    
    data_folder = args.data_folder
    add_promoter = args.add_promoter
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
    
    # Set seed for everything. Important for reproducibilty see: https://www.kaggle.com/code/bminixhofer/deterministic-neural-networks-using-pytorch
    # seed_everything(445566) # For other model configuration
    # seed_everything(44) # For model with promoter
    seed_everything(4455) # For m6A_info: flag_channel, add_promoter: True, target: control
    if mode=="train":
        logging.info(f"Training mode")
        logging.info(f"m6A_info: {m6A_info}, add_promoter: {add_promoter}, target: dual_outputs")
        suffix = f"dual_outputs_m6_info-{m6A_info}_promoter-{add_promoter}_{suffix}"
        for i in range(1, 2): #5-folds SHOULD BE 6
            logging.info(f"Fold-{i}")
            seq_fasta_train_path = f"{data_folder}/motif_fasta_train_SPLIT_{i}.fasta"
            # meta_data_train_json_path = f"{data_folder}/train_label_SPLIT_{i}.json"
            meta_data_train_json_path = f"{data_folder}/train_meta_data_SPLIT_{i}.json"
            m6A_info_train_path = None

            seq_fasta_test_path = f"{data_folder}/motif_fasta_test_SPLIT_{i}.fasta"
            # meta_data_test_json_path = f"{data_folder}/test_label_SPLIT_{i}.json"
            meta_data_test_json_path = f"{data_folder}/test_meta_data_SPLIT_{i}.json"
            m6A_info_test_path = None

            # add_promoter BOOL,
            promoter_fasta_train_path = None 
            promoter_fasta_test_path = None
            if add_promoter:
                promoter_fasta_train_path = f"{data_folder}/promoter_fasta_train_SPLIT_{i}.fasta"
                promoter_fasta_test_path = f"{data_folder}/promoter_fasta_test_SPLIT_{i}.fasta"

            # m6A_info ['flag_channel', 'level_channel', 'add_middle', 'no']
            # middle should come from the meta_data
            m6A_info_train_path = None
            m6A_info_test_path = None

            train_dataset = SequenceDatasetDual(seq_fasta_path=seq_fasta_train_path, meta_data_path=meta_data_train_json_path, prom_seq_fasta_path=promoter_fasta_train_path, m6A_info=m6A_info, m6A_info_path=m6A_info_train_path)
            test_dataset = SequenceDatasetDual(seq_fasta_path=seq_fasta_test_path, prom_seq_fasta_path=promoter_fasta_test_path,  meta_data_path=meta_data_test_json_path, m6A_info=m6A_info, m6A_info_path=m6A_info_test_path)
       
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

            dual_outputs = True

            input_size = 1001
            if add_promoter:
                input_size = 2001
            logging.info(f"Input size: {input_size}")
            config = {'cnn_first_filter': 10, 'cnn_first_kernel_size': 9, 'cnn_length': 2, 'cnn_filter': 64, 'cnn_kernel_size': 5, 'bilstm_layer': 3, 'bilstm_hidden_size': 64, 'fc_size': 256}
            if m6A_info=="level_channel" or m6A_info=="flag_channel": 
                model = NaiveModelV2(input_channel=5, cnn_first_filter=8, input_size=input_size)
                # model = ConvTransformerModel(input_channel=5)
                # model = MultiRMModel(1, True)
            else:
                # model = NaiveModelV2(input_channel=4, cnn_first_filter=8, input_size=input_size, output_dim=2)
                # model = ConvTransformerModel(input_channel=4)
                # model = MultiRMModel(1, True)
                model = ConfigurableModel(cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                              cnn_length=config["cnn_length"], cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                              output_size=2)

            model.to(device)
            #model=torch.nn.DataParallel(model) 


            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # here added the weight dacay
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # here added the weight dacay
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5) # here added the weight dacay for temp_w_l2reg. for temp no.
            #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.1) # here added exponenetial dcay for the optimizer
            # Train and Validate the model
            _, pred_true = train(model=model, train_loader=train_loader, test_loader=test_loader, epochs=num_epochs, loss_fn=loss_fn, save_dir=save_dir, learning_rate=learning_rate, device=device, optimizer=optimizer, suffix=f"{i}th_fold_{suffix}")

            model_file = f"{save_dir}/models/trained_model_{i}th_fold_{suffix}.pkl"
            torch.save(model.state_dict(), model_file)
            logging.info(f"Model of {i}th fold saved to {model_file}")

            if plot:
                plot_loss_function(f"{save_dir}/logs/training_{i}th_fold_{suffix}.log", f"{save_dir}/analysis", f"loss_plot_{i}th_fold_{suffix}")
                # plot_correlation(pred_true["true"], pred_true["pred"], f"{save_dir}/analysis", f"correlation_plot_{i}th_fold_{suffix}")

            save_val = True
            if save_val:
                with open(f"{save_dir}/validation_{i}th_fold_{suffix}.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_true["true"], pred_true["pred"]))
            logging.info(f"Finished on Fold-{i}")
    else:
        # TODO: Implement testing mode using the pickled model
        logging.info(f"Loading test data ")