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
from wrapper.data_setup import SequenceDatasetDual
from torchmetrics.wrappers import MultioutputWrapper
from torchmetrics.regression import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError
from wrapper.utils import seed_everything
from torch.utils.data import DataLoader
from wrapper.utils import EarlyStopper
import logging


def train_seq(config): 
    model = ConfigurableModel(cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                              cnn_length=config["cnn_length"], cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                              output_size=2)

    seed_everything(4455)                
    target = "case"
    i = 1
    epochs = 30
    data_folder = "/binf-isilon/renniegrp/vpx267/ucph_thesis/data/dual_outputs"
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

    train_dataset = SequenceDatasetDual(seq_fasta_path=seq_fasta_train_path, meta_data_path=meta_data_train_json_path, prom_seq_fasta_path=promoter_fasta_train_path, m6A_info=m6A_info_train_path, m6A_info_path=m6A_info_train_path)
    test_dataset = SequenceDatasetDual(seq_fasta_path=seq_fasta_test_path, prom_seq_fasta_path=promoter_fasta_test_path,  meta_data_path=meta_data_test_json_path, m6A_info=m6A_info_test_path, m6A_info_path=m6A_info_test_path)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=3)
    
    early_stopper = EarlyStopper(patience=3, min_delta=0.01)   
    for epoch in range(1, epochs+1):    
        model.train()
        train_loss = 0.0
        # Iterate in batches
            
        for batch, data in enumerate(train_loader):
            seq = data["seq"].to(device, non_blocking=True)
            meth_true_val = torch.stack([data[f"meth_control"].to(device, non_blocking=True), 
                                            data[f"meth_case"].to(device, non_blocking=True)], dim=1)
            
            optimizer.zero_grad(set_to_none=True)
            meth_pred_val = model(seq)
            loss1 = criterion(meth_pred_val[:,0], meth_true_val[:,0])
            loss2 = criterion(meth_pred_val[:,1], meth_true_val[:,1])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Average loss per batch in an epoch
        train_loss /= len(train_loader)


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
                loss1 = criterion(meth_pred_val[:,0], meth_true_val[:,0])
                loss2 = criterion(meth_pred_val[:,1], meth_true_val[:,1])
                loss = loss1 + loss2

                pred = torch.cat([pred, meth_true_val.cpu().detach()])
                true = torch.cat([true, meth_pred_val.cpu().detach()])
                
                # Total of average loss from each samples in a batch. Huberloss reduction is mean by default
                test_loss += loss.item()

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

        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.7f}, \n val_loss: {test_loss:.7f} \n val_control_mse:{metrics['mse'][0].item():.7f}, val_control_rmse: {metrics['rmse'][0].item():.7f}, val_control_mae: {metrics['mae'][0].item():.7f}, val_control_pearson_corr: {metrics['pearson_corr'][0].item():.7f} \n val_case_mse:{metrics['mse'][1].item():.7f}, val_case_rmse: {metrics['rmse'][1].item():.7f}, val_case_mae: {metrics['mae'][1].item():.7f}, val_case_pearson_corr: {metrics['pearson_corr'][1].item():.7f}")

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": test_loss, "val_control_mse":metrics['mse'][0], "val_case_mse":metrics['mse'][1], "val_control_pearson_corr": metrics['pearson_corr'][0], "val_case_pearson_corr": metrics['pearson_corr'][1]},
                checkpoint=checkpoint,
            )


def main(num_samples=10, max_num_epochs=50, gpus_per_trial=3):
    seed_everything(4455)
    config = {
        # Uniform distribuitino
        "lr": tune.choice([0.5e-2, 1e-3, 1e-2]),
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
            max_concurrent_trials=4,
        ),
        param_space=config,
        run_config=RunConfig(storage_path="/binf-isilon/renniegrp/vpx267/ucph_thesis/ray_results/adam", name="test_experiment")
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print(f"Best trial final validation control pearson correlation: {best_result.metrics['val_control_pearson_corr']}, case pearson correlation: {best_result.metrics['val_case_pearson_corr']}")

main(num_samples=50, max_num_epochs=15, gpus_per_trial=1)