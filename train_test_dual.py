import torch 
from torch.utils.data import DataLoader
import logging 
import os
import csv
from torchmetrics.wrappers import MultioutputWrapper
from torchmetrics.regression import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError, R2Score
from wrapper.utils import EarlyStopper
from tqdm import tqdm
from wrapper.utils import create_tensorboard_log_writer, extract_lstm_info

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

            pred = torch.cat([pred, meth_pred_val.cpu().detach()])
            true = torch.cat([true, meth_true_val.cpu().detach()])
            
            # Total of average loss from each samples in a batch. Huberloss reduction is mean by default
            test_loss += loss.item()

        # Average loss per batch in an epoch
        test_loss /= len(test_loader)

        pearson_corr = MultioutputWrapper(PearsonCorrCoef(), 2)
        mse = MultioutputWrapper(MeanSquaredError(), 2)
        mae = MultioutputWrapper(MeanAbsoluteError(), 2)
        R2 = MultioutputWrapper(R2Score(), 2)
        
        pearson_corr = pearson_corr(pred, true)
        mse = mse(pred, true)
        mae = mae(pred, true)
        rmse = torch.sqrt(mse)
        R2 = R2(pred, true)
        metrics = {"rmse": rmse.numpy(), 
                   "mse": mse.numpy(),
                   "mae": mae.numpy(),
                   "pearson_corr": pearson_corr.numpy(),
                   "pred": pred.numpy(),
                   "true": true.numpy(),
                   "R2": R2.numpy()}
    
    return test_loss, metrics    

def train(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
          loss_fn: torch.nn.Module, save_dir: str, target: None|str = "case", learning_rate: float=1e-3, 
          device: str="cpu", optimizer: torch.nn.Module|None=None, scheduler: torch.optim.lr_scheduler.LRScheduler|None=None, weighted_loss: bool=False, 
          tensorboard_writer: torch.utils.tensorboard.writer.SummaryWriter|None=None, patience: int|None=None, suffix: str="", **kwargs) -> tuple[torch.nn.Module, dict[str, float]]:
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
    param: tensorboard_writer: tensorboard writer for logging
    return: tuple[torch.nn.Module, dict[str, float]]
    """
    logging.info(f"==== Start training ====")
    logfile = open(f"{save_dir}/logs/training_{suffix}.log", "w+")
    logger = csv.DictWriter(logfile, fieldnames=["epoch", "train_loss", "val_control_loss", "val_control_rmse", "val_control_mse", "val_control_mae", "val_control_pearson_corr", "val_control_R2", "val_case_loss", "val_case_rmse", "val_case_mse", "val_case_mae", "val_case_pearson_corr", "val_case_R2"])
    logger.writeheader()
    
    if patience:
        early_stopper = EarlyStopper(patience=patience, min_delta=0.0001)    
    
    for epoch in range(1, epochs+1):    
        model.train()
        train_loss = 0.0
            
        epoch_iterator = tqdm(train_loader, desc="Train Loader Iteration")
        for batch, data in enumerate(epoch_iterator):
            seq = data["seq"].to(device, non_blocking=True)
            meth_true_val = torch.stack([data[f"meth_control"].to(device, non_blocking=True), 
                                            data[f"meth_case"].to(device, non_blocking=True)], dim=1)
            
            model.zero_grad(set_to_none=True) # safer than optimizer.zero_grad()
            meth_pred_val = model(seq)
            control_weight = data["meth_control_weight"].to(device, non_blocking=True)
            case_weight = data["meth_case_weight"].to(device, non_blocking=True)

            if not weighted_loss:
                control_weight = torch.ones_like(control_weight)
                case_weight = torch.ones_like(case_weight)

            loss1 = loss_fn(meth_pred_val[:,0], meth_true_val[:,0], control_weight)
            loss2 = loss_fn(meth_pred_val[:,1], meth_true_val[:,1], case_weight)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            # Total of average loss from each samples in a batch. Huberloss reduction is mean by default
            train_loss += loss.item()
        # Average loss per batch in an epoch
        train_loss /= len(train_loader)
        # Run scheduler if provided
        if scheduler:
            scheduler.step()

        validation_loss, metrics = test(model=model, test_loader=test_loader, loss_fn=loss_fn, 
                                        target=target, device=device)

        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.7f}, \n val_loss: {validation_loss:.7f} \n val_control_mse:{metrics['mse'][0].item():.7f}, val_control_rmse: {metrics['rmse'][0].item():.7f}, val_control_mae: {metrics['mae'][0].item():.7f}, val_control_pearson_corr: {metrics['pearson_corr'][0].item():.7f}, val_control_R2: {metrics['R2'][0].item():.7f} \n val_case_mse:{metrics['mse'][1].item():.7f}, val_case_rmse: {metrics['rmse'][1].item():.7f}, val_case_mae: {metrics['mae'][1].item():.7f}, val_case_pearson_corr: {metrics['pearson_corr'][1].item():.7f}, val_case_R2: {metrics['R2'][1].item():.7f}")
        logger.writerow({"epoch": epoch, "train_loss": train_loss, "val_control_mse":metrics['mse'][0], "val_control_rmse": metrics['rmse'][0], "val_control_mae": metrics['mae'][0], "val_control_pearson_corr": metrics['pearson_corr'][0],"val_control_R2":metrics['R2'][0],"val_case_mse":metrics['mse'][1], "val_case_rmse": metrics['rmse'][1], "val_case_mae": metrics['mae'][1], "val_case_pearson_corr": metrics['pearson_corr'][1],"val_case_R2":metrics['R2'][1]})

        fold_info = ""
        if kwargs["fold_info"]:
            fold_info = f"{kwargs['fold_info']}/"

        if tensorboard_writer:
            # Logging grads
            for name, param in model.named_parameters():
                    tensorboard_writer.add_histogram(
                        tag=f"{fold_info}run_params/{name}", values=param, global_step=epoch
                    )
                    tensorboard_writer.add_histogram(
                        tag=f"{fold_info}run_grads/{name}", values=param.grad, global_step=epoch
                    )
            
            # Logging performance
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}Loss/Train", 
                               tag_scalar_dict={"train_loss": train_loss}, 
                               global_step=epoch)
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}Loss/Validation", 
                               tag_scalar_dict={"validation_loss": validation_loss}, 
                               global_step=epoch)
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}MSE/Control", 
                               tag_scalar_dict={"val_control_mse": metrics['mse'][0].item()}, 
                               global_step=epoch)
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}MSE/Case", 
                               tag_scalar_dict={"val_case_mse": metrics['mse'][1].item()}, 
                               global_step=epoch)
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}Pearson_Correlation/Control", 
                               tag_scalar_dict={"val_control_pearson_corr": metrics['pearson_corr'][0].item()}, 
                               global_step=epoch)
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}Pearson_Correlation/Case", 
                               tag_scalar_dict={"val_case_pearson_corr": metrics['pearson_corr'][1].item()}, 
                               global_step=epoch)
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}R2/Control", 
                               tag_scalar_dict={"val_control_pearson_corr": metrics['R2'][0].item()}, 
                               global_step=epoch)
            tensorboard_writer.add_scalars(main_tag=f"{fold_info}R2/Case", 
                               tag_scalar_dict={"val_case_pearson_corr": metrics['R2'][1].item()}, 
                               global_step=epoch)
            tensorboard_writer.close()
        
        if patience:
            if early_stopper.early_stop(validation_loss):        
                logging.info(f"Early stop at epoch: {epoch}")     
                break

    plts = extract_lstm_info(model)
    if plts:
        for name, plt in plts.items(): 
            tensorboard_writer.add_figure(f"{fold_info}LSTM Weight/{name}", plt.gcf())
        tensorboard_writer.close()
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
    python train_test_dual.py --data_folder data/dual_outputs --num_epochs 200 --batch_size 256 --save_dir data/outputs --mode train --embedding gene2vec --embedding_file data/embeddings/gene2vec/double_outputs --learning_rate 0.001 --gpu 5 --loader_worker 4  --suffix TEST_DUAL_OUTPUTS_BEST_PARAMS_PROMOTER
    
    with m6A channel
    -------
    python train_test_dual.py --data_folder data/dual_outputs --num_epochs 200 --batch_size 256 --save_dir data/outputs --mode train --learning_rate 0.001 --gpu 5 --add_promoter y  --loader_worker 4  --suffix TEST_DUAL_OUTPUTS_BEST_PARAMS_PROMOTER
    """

    from argparse import ArgumentParser, ArgumentTypeError
    from wrapper.data_setup import SequenceDatasetDual, SequenceDatasetDualGene2Vec, SequenceDatasetDualFilter, SequenceDatasetDualShortenedFeatures
    from wrapper.utils import plot_loss_function, plot_correlation, seed_everything, BMCLoss
    from torchinfo import summary
    import wrapper.weighted_losses as wloss
    from model import NaiveModelV1, NaiveModelV2, NaiveModelV3, MultiRMModel, ConvTransformerModel, ConfigurableModelWoBatchNorm, TestMotifModel, TestMotifModel2, TestMotifModel3, LSTMOnly, TestMotifModelWithSelfAttention, TestMotifModelWithSelfAttention2, AttentionOnly, TestMotifModelBranchedEnd, TestMotifModelDropoutTest
    import sys
    import subprocess

    # Set logging template
    logging.basicConfig(format='%(asctime)s::%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', 
                        level = logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
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
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")   
    parser.add_argument("--gpu", default=[0,1], nargs='+', type=int)
    parser.add_argument("--loader_worker", default=0, type=int)
    parser.add_argument("--add_promoter", type=str2bool, nargs='?', const=True, default=False, help="Append promoter sequence[y/n]")
    parser.add_argument("--m6A_info", default='no',const='no', nargs='?', 
                        choices=['flag_channel', 'level_channel', 'add_prob_middle', 'add_flag_middle', 'no'], 
                        help="Include m6A info? ['flag_channel', 'level_channel', 'add_middle', 'no']")
    parser.add_argument("--embedding", default='one-hot',const='one-hot', nargs='?', 
                        choices=['one-hot', 'gene2vec'], 
                        help="Embedding options ['one-hot', 'gene2vec']")
    parser.add_argument("--embedding_file", default=None, help="Embedding file for gene2vec")
    #parser.add_argument("--m6A_info", type=str2bool, nargs='?', const=True, default=False, help="Include m6A info? [y/n]")
    parser.add_argument("--plot", type=str2bool, nargs='?', const=True, default=False, help="Plot loss or not [y/n]")
    parser.add_argument("--suffix", default="", help="Suffix for output files")
    parser.add_argument("--patience", help="Number of patience for training", default=2, type=int)
    parser.add_argument("--weighted_loss", type=str2bool, nargs='?', default=False, help="Weighted Loss[y/n]")
    parser.add_argument("--tensorboard_writer", type=str2bool, nargs='?', default=True, help="Tensor Board Logging[y/n]")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)    
    num_epochs = int(args.num_epochs)
    loader_worker = int(args.loader_worker)


    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv']).decode().split('\n')[1]
        logging.info(f"Running on device: {device} | GPU name: {gpu_name}")  
    else:
        device = torch.device("cpu")
        logging.info(f"Running on device: {device}")  
    
    # Set seed for everything. Important for reproducibilty see: https://www.kaggle.com/code/bminixhofer/deterministic-neural-networks-using-pytorch
    seed_everything(1234)
    if args.mode=="train":
        logging.info(f"Training mode")
        logging.info(f"m6A_info: {args.m6A_info}, add_promoter: {args.add_promoter}, embedding: {args.embedding}, target: dual_outputs")
        suffix = f"dual_outputs_m6A_info-{args.m6A_info}_promoter-{args.add_promoter}_{args.suffix}"

        folds = [1,2,3,4,5]
        for fold in folds:
            logging.info(f"Fold-{fold}")
            seq_fasta_train_path = f"{args.data_folder}/motif_fasta_train_SPLIT_{fold}.fasta"
            # meta_data_train_json_path = f"{args.data_folder}/train_label_SPLIT_{fold}.json"
            meta_data_train_json_path = f"{args.data_folder}/train_meta_data_SPLIT_{fold}.json"
            m6A_info_train_path = None

            seq_fasta_test_path = f"{args.data_folder}/motif_fasta_test_SPLIT_{fold}.fasta"
            # meta_data_test_json_path = f"{args.data_folder}/test_label_SPLIT_{fold}.json"
            meta_data_test_json_path = f"{args.data_folder}/test_meta_data_SPLIT_{fold}.json"
            m6A_info_test_path = None

            promoter_fasta_train_path = None 
            promoter_fasta_test_path = None
            if args.add_promoter:
                promoter_fasta_train_path = f"{args.data_folder}/promoter_fasta_train_SPLIT_{fold}.fasta"
                promoter_fasta_test_path = f"{args.data_folder}/promoter_fasta_test_SPLIT_{fold}.fasta"

            # m6A_info ['flag_channel', 'level_channel', 'add_middle', 'no']
            # middle should come from the meta_data
            m6A_info_train_path = None
            m6A_info_test_path = None

            embedding_file = None
            if args.embedding_file:
                embedding_file = f"{args.embedding_file}/split_{fold}.model"

            logging.info(f"Loading SequenceDataset")
            train_dataset = SequenceDatasetDualShortenedFeatures(seq_fasta_path=seq_fasta_train_path, meta_data_path=meta_data_train_json_path, prom_seq_fasta_path=promoter_fasta_train_path, m6A_info=args.m6A_info, m6A_info_path=m6A_info_train_path, transform=args.embedding, path_to_embedding=embedding_file)
            test_dataset = SequenceDatasetDualShortenedFeatures(seq_fasta_path=seq_fasta_test_path, prom_seq_fasta_path=promoter_fasta_test_path,  meta_data_path=meta_data_test_json_path, m6A_info=args.m6A_info, m6A_info_path=m6A_info_test_path, transform=args.embedding, path_to_embedding=embedding_file)

            # USE BIG WORKER FOR THIS i.e. 10 worker for loader
            # path_to_seq = "data/dual_outputs/motif_fasta_train_SPLIT_1.fasta"
            # hdf_file = "/binf-isilon/renniegrp/vpx267/ucph_thesis/data/dual_outputs/hdf5/gene2vec.hdf5"
            # path_to_prom = "data/dual_outputs/promoter_fasta_test_SPLIT_1.fasta"
            # path_to_meta_data_train = "data/dual_outputs/train_meta_data_SPLIT_1.json"
            # path_to_meta_data_test = "data/dual_outputs/test_meta_data_SPLIT_1.json"
            # path_to_embedding = "data/embeddings/gene2vec/dual_outputs/split_1.model"
            # train_dataset = SequenceDatasetDualGene2Vec(hdf_file, dataset="train/motif_SPLIT_1",  meta_data_path=path_to_meta_data_train)
            # test_dataset = SequenceDatasetDualGene2Vec(hdf_file, dataset="test/motif_SPLIT_1",  meta_data_path=path_to_meta_data_test)



            logging.info(f"Loading Dataloader")
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=loader_worker, persistent_workers=True, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=loader_worker, persistent_workers=True, pin_memory=True)
            logging.info(f"Number of training batches: {len(train_loader)}, Total training sample: {len(train_loader.dataset)}")
            logging.info(f"Number of test batches: {len(test_loader)}, Total test sample: {len(test_loader.dataset)}")

            
            # 0.5MSE if delta<0.1, otherwise delta(|y-y_hat| - 0.5delta)
            # loss_fn = torch.nn.HuberLoss(delta=1)
            # loss_fn = torch.nn.MSELoss()
            # loss_fn = BMCLoss()
            loss_fn = wloss.weighted_mse_loss

            dual_outputs = True
            input_size = train_dataset.seq.shape[2]
            logging.info(f"Input size: {input_size}")
            # lr = 0.01
            config = {'cnn_first_filter': 16, 'cnn_first_kernel_size': 9, 'cnn_length': 3, 'cnn_filter': 32, 'cnn_kernel_size': 7, 'bilstm_layer': 3, 'bilstm_hidden_size': 128, 'fc_size': 64}
            
            if args.embedding=="one-hot":
                if args.m6A_info=="level_channel" or args.m6A_info=="flag_channel": 
                    # model = NaiveModelV2(input_channel=5, cnn_first_filter=8, input_size=input_size)
                    # model = ConvTransformerModel(input_channel=5)
                    # model = MultiRMModel(1, True)
                    # model = ConfigurableModelWoBatchNorm(input_channel=5, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                                # cnn_length=config["cnn_length"], cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                                # output_size=2)
                    # model = LSTMOnly(input_channel=5, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                    #             cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                    #             output_size=2)
                    # model = TestMotifModelWithSelfAttention(input_channel=5, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                    #             cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], encoder_head=8, num_encoder_layer=3, encoder_dim_feedforward=1024, 
                    #             output_size=2)
                    model = TestMotifModelBranchedEnd(input_channel=5, input_size=input_size, cnn_first_filter=config["cnn_first_filter"], 
                                                      cnn_first_kernel_size=config["cnn_first_kernel_size"],
                                                      cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],output_size=2)
                    # model = AttentionOnly(input_channel=5, encoder_head=5, num_encoder_layer=6, encoder_dim_feedforward=2048, output_size=2)
                else:
                    # model = NaiveModelV2(input_channel=4, cnn_first_filter=8, input_size=input_size, output_dim=2)
                    # model = ConvTransformerModel(input_channel=4)
                    # model = MultiRMModel(1, True)
                    # model = ConfigurableModelWoBatchNorm(input_channel=4, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                    #             cnn_length=config["cnn_length"], cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                    #             output_size=2)
                    # model = LSTMOnly(input_channel=4, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                    #             cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                    #             output_size=2)
                    # model = TestMotifModelWithSelfAttention(input_channel=4, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                    #             cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], encoder_head=8, num_encoder_layer=3, encoder_dim_feedforward=1024, 
                    #             output_size=2)
                    # model = AttentionOnly(input_channel=4, encoder_head=4, num_encoder_layer=6, encoder_dim_feedforward=2048, output_size=2)

                    # model = TestMotifModel(input_channel=4, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                    #                         cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                    #                         output_size=2)
                    model = TestMotifModel(input_channel=4, input_size=input_size, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                                            cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                                            output_size=2)

            if args.embedding=="gene2vec":
                model = ConfigurableModelWoBatchNorm(input_channel=300, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                            cnn_length=config["cnn_length"], cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                            output_size=2)
            
            tensorboard_writer = None 
            if args.tensorboard_writer:
                tensorboard_writer = create_tensorboard_log_writer(experiment_name=f"{suffix}", model_name=model.__class__.__name__, log_dir=f"{args.save_dir}/tensorboard_logs")
            
            model.to(device)
            #model=torch.nn.DataParallel(model) 

            if fold == folds[0]:
                # Print summary on the first fold
                summary(model, verbose=1, col_width=15, input_size=(args.batch_size, 4, input_size),  col_names=["input_size", "output_size", "num_params",  "params_percent", "trainable"])
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2), weight_decay=0.1) 
            # Train and Validate the model
            _, pred_true = train(model=model, train_loader=train_loader, test_loader=test_loader, epochs=num_epochs, loss_fn=loss_fn, save_dir=args.save_dir, learning_rate=args.learning_rate, device=device, optimizer=optimizer, tensorboard_writer=tensorboard_writer, weighted_loss=args.weighted_loss, patience=args.patience, suffix=f"{fold}th_fold_{suffix}", fold_info=fold)

            model_file = f"{args.save_dir}/models/trained_model_{fold}th_fold_{suffix}.pkl"
            torch.save(model.state_dict(), model_file)
            logging.info(f"Model of {fold}th fold saved to {model_file}")

            if args.plot:
                plot_loss_function(f"{args.save_dir}/logs/training_{fold}th_fold_{suffix}.log", f"{args.save_dir}/analysis", f"loss_plot_{fold}th_fold_{suffix}")
                plot_correlation(pred_true["true"][:,0], pred_true["pred"][:,0], f"{args.save_dir}/analysis", f"correlation_plot_{fold}th_fold_{suffix}", "CONTROL")
                plot_correlation(pred_true["true"][:,1], pred_true["pred"][:,1], f"{args.save_dir}/analysis", f"correlation_plot_{fold}th_fold_{suffix}", "CASE")

            save_val = True
            if save_val:
                with open(f"{args.save_dir}/predictions/validation_{fold}th_fold_{suffix}.csv", "w") as f:
                    writer = csv.writer(f) 
                    writer.writerow(["true_control","true_case","pred_control","pred_case"])
                    for true, pred in zip(pred_true["true"], pred_true["pred"]):
                        writer.writerow(list(true) + list(pred))
            logging.info(f"Finished on Fold-{fold}")
    else:
        # TODO: Implement testing mode using the pickled model
        logging.info(f"Loading test data ")