import torch 
import numpy as np
from torch.utils.data import DataLoader
import logging 
from argparse import ArgumentParser, ArgumentTypeError
import os
import csv
from datetime import datetime


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
        rmse = 0.0
        for data in test_loader:
            seq = data["seq"].to(device)
            prom_seq = data["prom_seq"].to(device)
            meth_control = data["meth_control"].to(device)
            meth_case = data["meth_case"].to(device)
            

            meth_pred_response = model(seq, prom_seq, meth_control)

            loss = loss_fn(meth_pred_response, meth_case)
            test_loss += loss.item()

            for i in range(len(meth_pred_response)):
                y_pred = meth_pred_response[i].cpu().detach().numpy()
                y_true = meth_case[i].cpu().detach().numpy()

                rmse += (y_pred - y_true)**2

        # Total RMSE
        rmse = np.sqrt(rmse/len(test_loader.dataset)) 
        test_loss /= len(test_loader.dataset)
        metrics = {"rmse": rmse}

        logging.info(f"Test Loss: {test_loss}, RMSE: {rmse}")
    
    return test_loss, metrics    

def train(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
          loss_fn: torch.nn.Module, save_dir: str, learning_rate: float=1e-3, 
          device: str="cpu", optimizer: torch.nn.Module|None=None):
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
    
    logfile = open(f"{save_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M")}.log", "w")
    logger = csv.DictWriter(logfile, fieldnames=["epoch", "train_loss", "val_loss", "val_rmse"])

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in epochs:    
        model.train()
        train_loss = 0.0
        # Iterate in batches
        for batch, data in enumerate(train_loader):
            seq = data["seq"].to(device)
            prom_seq = data["prom_seq"].to(device)
            meth_case = data["meth_case"].to(device)
            meth_control = data["meth_control"].to(device)
            
            optimizer.zero_grad()
            meth_pred_response = model(seq, prom_seq, meth_control)
            loss = loss_fn(meth_pred_response, meth_case)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)

        validation_loss, metrics = test(mode, test_loader, loss_fn, device)

        logging.info(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, Validation RMSE: {metrics['rmse']}")
        logger.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss, "val_rmse": metrics["rmse"]})
    
    logfile.close()

    logging.info(f"==== End training =====")
    return model

if __name__=="__main__":
    """
    EXAMPLE USAGE:
    -------
    python train_test.py --data_folder /Users/faqih/Documents/UCPH/Thesis/code/data/train_test_data --num_epochs 10 --batch_size 100 --save_dir /Users/faqih/Documents/UCPH/Thesis/code/data/outputs --mode train --learning_rate 0.001    
    """

    from wrapper.data_setup import SequenceDataset
    from model import ModelV1
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
    parser.add_argument("-F", "--data_folder", help="Train test Data Folder", required=True)
    parser.add_argument("-E","--num_epochs", help="Number of epochs", required=True)
    parser.add_argument("-B","--batch_size", help="Number of batch size", default=100, type=int)
    parser.add_argument("-S","--save_dir", help="Directory for saving outputs", required=True)
    parser.add_argument("-M","--mode", help="Training or testing mode",required=True, 
                        choices=["train","test"])
    parser.add_argument("-L","--learning_rate", help="Learning rate", default=0.001, type=float)
    args = parser.parse_args()


    
    data_folder = args.data_folder
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    save_dir = args.save_dir
    mode = args.mode
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
            label_train_json_path = f"{data_folder}/train_label_SPLIT_{i}.json"

            seq_fasta_test_path = f"{data_folder}/motif_fasta_test_SPLIT_{i}.fasta"
            promoter_fasta_test_path = f"{data_folder}/promoter_fasta_test_SPLIT_{i}.fasta"
            label_test_json_path = f"{data_folder}/test_label_SPLIT_{i}.json"

            train_dataset = SequenceDataset(seq_fasta_train_path, promoter_fasta_train_path, label_train_json_path)
            test_dataset = SequenceDataset(seq_fasta_test_path, promoter_fasta_test_path, label_test_json_path)

            print(f"Train Dataset Object: {test_dataset}")
            print(train_dataset)
            print(f"Train seq shape: {train_dataset.seq.shape}, type: {type(train_dataset.seq)}")
            print(train_dataset.seq)
            print(f"Train label shape: {train_dataset.label.shape}, type: {type(train_dataset.label)}")
            print(train_dataset.label)
            print(f"Train prom_seq shape: {train_dataset.prom_seq.shape}, type: {type(train_dataset.prom_seq)}")
            print(train_dataset.prom_seq)

            print(f"Test Dataset Object: {test_dataset}")
            print(f"Test seq shape: {test_dataset.seq.shape}, type: {type(test_dataset.seq)}")
            print(test_dataset.seq)
            print(f"Test label shape: {test_dataset.label.shape}, type: {type(test_dataset.label)}")
            print(test_dataset.label)
            print(f"Test prom_seq shape: {test_dataset.prom_seq.shape}, type: {type(test_dataset.prom_seq)}")
            print(test_dataset.prom_seq)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            print(f"Train Loader Object {train_loader}")
            print(f"Test Loader Object {test_loader}")

            # loss_fn = torch.nn.HuberLoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # model = ModelV1()
            # mode.to(device)
            # # Train and Validate the model
            # train(model, train_loader, test_loader, num_epochs, loss_fn, save_dir, learning_rate, device, optimizer)

            # model_file = f'{save_dir}/trained_model_{i}th_fold.pkl'
            # torch.save(model.state_dict(), model_file)
            # logging.info(f"Model of {i}th fold saved to {model_file}")
            logging.info(f"Finished on Fold-{i}")
    else:
        # TODO: Implement testing mode using the pickled model
        pass