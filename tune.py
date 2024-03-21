from typing import Dict
import ray
import os
import numpy as np
import torch
import tempfile
import torch.nn as nn
import torch.optim as optim
from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from model import ConfigurableModel
from wrapper.data_setup import SequenceDataset
from wrapper.utils import seed_everything
from torch.utils.data import DataLoader
from wrapper.utils import EarlyStopper
from torchmetrics.functional.regression import pearson_corrcoef
import logging


def train_seq(config): 
    model = ConfigurableModel(cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                              cnn_length=config["cnn_length"], cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"])

    seed_everything(4455)                
    target = "case"
    i = 1
    epochs = 30
    data_folder = "/binf-isilon/renniegrp/vpx267/ucph_thesis/data/single_model/case"
    add_promoter = False 
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


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.HuberLoss(delta=1)
    # optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

            
    

    train_dataset = SequenceDataset(seq_fasta_path=seq_fasta_train_path, meta_data_path=meta_data_train_json_path, prom_seq_fasta_path=promoter_fasta_train_path, m6A_info_path=m6A_info_train_path, target=target)
    test_dataset = SequenceDataset(seq_fasta_path=seq_fasta_test_path, prom_seq_fasta_path=promoter_fasta_test_path,  meta_data_path=meta_data_test_json_path, m6A_info_path=m6A_info_test_path, target=target)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    early_stopper = EarlyStopper(patience=3, min_delta=0.01)   
    for epoch in range(1, epochs+1):    
        model.train()
        train_loss = 0.0
        # Iterate in batches

        # logging.info(f"Starting batch")
            
        for batch, data in enumerate(train_loader):
            seq = data["seq"].to(device, non_blocking=True)
            meth_true_val = data[f"meth_{target}"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            meth_pred_val = model(seq)
            loss = criterion(meth_pred_val.squeeze(1), meth_true_val)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Average loss per batch in an epoch
        train_loss /= len(train_loader)


        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            se = 0.0
            ae = 0.0
            pred = torch.Tensor().to("cpu", non_blocking=True)
            true = torch.Tensor().to("cpu", non_blocking=True)
            for data in test_loader:
                seq = data["seq"].to(device, non_blocking=True)
                meth_true_val = data[f"meth_{target}"].to(device, non_blocking=True)

                meth_pred_val = model(seq)
                loss = criterion(meth_pred_val.squeeze(1), meth_true_val)

                pred = torch.cat([pred, meth_pred_val.squeeze(1).cpu().detach()])
                true = torch.cat([true, meth_true_val.cpu().detach()])
                
                # Total of average loss from each samples in a batch. Huberloss reduction is mean by default
                test_loss += loss.item()

                for i in range(len(meth_pred_val)):
                    y_pred = meth_pred_val[i].cpu().detach().numpy()
                    y_true = meth_true_val[i].cpu().detach().numpy()
                    
                    # squared and abs error
                    se += (y_pred - y_true)**2
                    ae += np.abs(y_pred - y_true)

            num_samples = len(test_loader.dataset)
            num_batches = len(test_loader)

            # Average loss per batch in an ecpoh
            test_loss /= num_batches

            # metrics
            mse =  (se/num_samples).item() 
            rmse = (np.sqrt(se/num_samples)).item()
            mae = (ae/num_samples).item()

            #logging.info(f"Pred size: {pred.shape}, true size: {true.shape}")
            # Pearson correlation for all of pred and true 
            # x100 for numerical stability
            pearson_corr = pearson_corrcoef(pred, true).item()
            metrics = {"rmse": rmse, 
                    "mse": mse,
                    "mae": mae,
                    "pearson_corr": pearson_corr,
                    "pred": pred.numpy(),
                    "true": true.numpy()}

        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.7f}, val_loss: {test_loss:.7f}, val_mse:{metrics['mse']:.7f}, val_rmse: {metrics['rmse']:.7f}, val_mae: {metrics['mae']:.7f}, val_pearson_corr: {metrics['pearson_corr']:.7f}")

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": test_loss, "pearson_corr": metrics["pearson_corr"]},
                checkpoint=checkpoint,
            )


def main(num_samples=10, max_num_epochs=50, gpus_per_trial=3):
    seed_everything(4455)
    config = {
        # Uniform distribuitino for choicing
        "lr": tune.choice([1e-4, 1e-3, 1e-2]),
        "cnn_first_filter": tune.choice([8, 10, 12]),
        "cnn_first_kernel_size": tune.choice([5,7,9]),
        "cnn_length": tune.choice([2, 3]),
        "cnn_filter": tune.choice([32, 64]),
        "cnn_kernel_size": tune.choice([3, 5, 7]),
        "bilstm_layer": tune.choice([2, 3]),
        "bilstm_hidden_size": tune.choice([128, 64]),
        "fc_size": tune.choice([64, 128, 256])
    }


    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_seq),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            reuse_actors=False,
            num_samples=num_samples,
            max_concurrent_trials=7,
        ),
        param_space=config,
        run_config=RunConfig(storage_path="/binf-isilon/renniegrp/vpx267/ucph_thesis/ray_results/adam", name="test_experiment")
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation pearson correlation: {}".format(
        best_result.metrics["pearson_corr"]))

main(num_samples=500, max_num_epochs=15, gpus_per_trial=1)