from typing import Dict
import os
import sys
import numpy as np
import torch
import tempfile
import torch.nn as nn
import torch.optim as optim
from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from model import ConfigurableModel
from wrapper.data_setup import SequenceDatasetDual, SequenceDatasetDualGene2Vec
from torchmetrics.wrappers import MultioutputWrapper
from torchmetrics.regression import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError
from wrapper.utils import seed_everything
from torch.utils.data import DataLoader
from wrapper.utils import EarlyStopper
import logging
from argparse import ArgumentParser

def train_seq(config, input_channel, input_size, train_loader, test_loader): 
    model = ConfigurableModel(input_channel=input_channel, input_size=input_size, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                              cnn_length=config["cnn_length"], cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                              output_size=2)

    seed_everything(1234)        
    epochs = 100 # set large enough for max_t of ASHAScheduler


    print("Initialize model, loss_fn and optimizer")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    print(f"Using device: {device}")
    model.to(device)
    criterion = nn.HuberLoss(delta=1)
    # optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    print("Finished initialize model, loss_fn and optimizer")
    
    # print("Initialize checkpoint")
    # if train.get_checkpoint():
    #     loaded_checkpoint = train.get_checkpoint()
    #     with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
    #         model_state, optimizer_state = torch.load(
    #             os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
    #         )
    #         model.load_state_dict(model_state)
    #         optimizer.load_state_dict(optimizer_state)
    # print("Finish initialize checkpoint")
    
    early_stopper = EarlyStopper(patience=3, min_delta=0.01)   
    print("Starting first epoch")
    for epoch in range(1, epochs+1):    
        model.train()
        train_loss = 0.0
        # Iterate in batches
        print("Starting training")    
        for batch, data in enumerate(train_loader):
            seq = data["seq"].to(device, non_blocking=True)
            meth_true_val = torch.stack([data[f"meth_control"].to(device, non_blocking=True), 
                                            data[f"meth_case"].to(device, non_blocking=True)], dim=1)
            
            model.zero_grad(set_to_none=True) # safer than optimizer.zero_grad()
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

        print(f"Epoch: {epoch}, train_loss: {train_loss:.7f}, \n val_loss: {test_loss:.7f} \n val_control_mse:{metrics['mse'][0].item():.7f}, val_control_rmse: {metrics['rmse'][0].item():.7f}, val_control_mae: {metrics['mae'][0].item():.7f}, val_control_pearson_corr: {metrics['pearson_corr'][0].item():.7f} \n val_case_mse:{metrics['mse'][1].item():.7f}, val_case_rmse: {metrics['rmse'][1].item():.7f}, val_case_mae: {metrics['mae'][1].item():.7f}, val_case_pearson_corr: {metrics['pearson_corr'][1].item():.7f}")

        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        #     path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        #     torch.save(
        #         (model.state_dict(), optimizer.state_dict()), path
        #     )
        #     checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        #     train.report(
        #         {"loss": test_loss, "val_control_mse":metrics['mse'][0], "val_case_mse":metrics['mse'][1], "val_control_pearson_corr": metrics['pearson_corr'][0], "val_case_pearson_corr": metrics['pearson_corr'][1]},
        #         checkpoint=checkpoint,
        #     )


def main(num_samples=10, max_num_epochs=50, gpus_per_trial=1, args=None):
    seed_everything(1234)
    config = {
        # Uniform distribuitinon
        "lr": tune.choice([0.5e-2, 1e-3, 0.5e-3, 1e-2]),
        "cnn_first_filter": tune.choice([8, 12, 16]), #  input and output channels to be divisible by 8 (for FP16) or 4 (for TF32) to run efficiently on Tensor Cores
        "cnn_first_kernel_size": tune.choice([5,7,9]),
        "cnn_length": tune.choice([2, 3]),
        "cnn_filter": tune.choice([32, 64]),
        "cnn_kernel_size": tune.choice([3, 5, 7]),
        "bilstm_layer": tune.choice([2, 3]),
        "bilstm_hidden_size": tune.choice([256, 128, 64]), # 8 (for FP16) or 4 (for TF32) to run efficiently on Tensor Cores AND Divisible by at least 64 and ideally 256 to improve tiling efficiency   
        "fc_size": tune.choice([64, 128, 256]) # Batch size and the number of inputs and outputs to be divisible by 4 (TF32) / 8 (FP16) / 16 (INT8) to run efficiently on Tensor Cores
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
            
    i = 1
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

    embedding_file = None
    if args.embedding_file:
        embedding_file = f"{args.embedding_file}/split_{i}.model"
    print("Generate Dataset")
    # # Just remove the embedding related argument to tune one-hot
    # train_dataset = SequenceDatasetDual(seq_fasta_path=seq_fasta_train_path, meta_data_path=meta_data_train_json_path, prom_seq_fasta_path=promoter_fasta_train_path, m6A_info=m6A_info_train_path, m6A_info_path=m6A_info_train_path, transform=args.embedding, path_to_embedding=embedding_file)
    # test_dataset = SequenceDatasetDual(seq_fasta_path=seq_fasta_test_path, prom_seq_fasta_path=promoter_fasta_test_path,  meta_data_path=meta_data_test_json_path, m6A_info=m6A_info_test_path, m6A_info_path=m6A_info_test_path, transform=args.embedding, path_to_embedding=embedding_file)
    
    # USE BIG WORKER FOR THIS
    path_to_seq = "data/dual_outputs/motif_fasta_train_SPLIT_1.fasta"
    hdf_file = "/binf-isilon/renniegrp/vpx267/ucph_thesis/data/dual_outputs/hdf5/gene2vec.hdf5"
    path_to_prom = "data/dual_outputs/promoter_fasta_test_SPLIT_1.fasta"
    path_to_meta_data_train = "data/dual_outputs/train_meta_data_SPLIT_1.json"
    path_to_meta_data_test = "data/dual_outputs/test_meta_data_SPLIT_1.json"
    path_to_embedding = "data/embeddings/gene2vec/dual_outputs/split_1.model"
    input_channel = 300
    input_size = 999
    train_dataset = SequenceDatasetDualGene2Vec(hdf_file, dataset="train/motif_SPLIT_1",  meta_data_path=path_to_meta_data_train)
    test_dataset = SequenceDatasetDualGene2Vec(hdf_file, dataset="test/motif_SPLIT_1",  meta_data_path=path_to_meta_data_test)
    print("Finished Generate Dataset")
    print("Starting dataloader setup")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=5)
    print("Finished dataloader setup")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_seq, input_channel=300, input_size=999, train_loader=train_loader, test_loader=test_loader),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            reuse_actors=False,
            num_samples=num_samples,
            max_concurrent_trials=5,
        ),
        param_space=config,
        run_config=RunConfig(storage_path=args.tune_output_path, name=args.tune_output_folder_name)
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print(f"Best trial final validation control pearson correlation: {best_result.metrics['val_control_pearson_corr']}, case pearson correlation: {best_result.metrics['val_case_pearson_corr']}")

if __name__ == "__main__":
    """
    EXAMPLE
    -----

    python tune_dual.py --embedding gene2vec --embedding_file data/embeddings/gene2vec/double_outputs --tune_output_path /binf-isilon/renniegrp/vpx267/ucph_thesis/ray_results --tune_output_folder_name gene2vec --num_samples 100 --max_num_epochs 15
    """
    parser = ArgumentParser(
        description="Running Ray Tuning"
    )
    parser.add_argument("--embedding", default='one-hot',const='one-hot', nargs='?', 
                        choices=['one-hot', 'gene2vec'], 
                        help="Embedding options ['one-hot', 'gene2vec']")
    parser.add_argument("--embedding_file", default=None, help="Embedding file for gene2vec")
    parser.add_argument("--tune_output_path", default="ray_results", help="Output path for tune")
    parser.add_argument("--tune_output_folder_name", default="tuning", help="Output folder name for tune")
    parser.add_argument("--num_samples", default=50, type=int, help="number of samples for ray tuning")
    parser.add_argument("--max_num_epochs", default=15, type=int, help="number of samples for ray tuning")
    args = parser.parse_args()

    main(num_samples=args.num_samples, max_num_epochs=args.max_num_epochs, gpus_per_trial=1, args=args)