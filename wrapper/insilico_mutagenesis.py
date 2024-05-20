import numpy as np
import torch
import copy
from multiprocessing import Pool
import matplotlib.pyplot as plt


def mutagenesis(x: torch.Tensor, model: torch.nn.Module, mutation_size :int=50, class_index: None|int=None, verbose: bool=True, batch_size = 1024, cuda_device: int=6) -> np.ndarray:
    """ 
    in silico mutagenesis

    input: x: one-hot-encoded sequences with shape of (batch, seq_length, 4)
    input: model: trained model 

    """
    device = (torch.device(f"cuda:{cuda_device}") if torch.cuda.is_available()
            else torch.device("cpu"))

    if device == torch.device(f"cuda:{cuda_device}"):
        torch.set_default_device(f"cuda:{cuda_device}")
        torch.cuda.set_device(f"cuda:{cuda_device}") 
        if not x.is_cuda:
            x = x.cuda(cuda_device)
        if next(model.parameters()).device == torch.device("cpu"): 
            model = model.cuda(cuda_device)
    
    model.eval()

    def generate_mutagenesis(x: np.ndarray, mutation_size: int) -> np.ndarray:
        """
        Assuming input sequence is odd, It will output single nucleotide mutation at each position of the input sequence except the middle position.
        It requires the mutation size to be even.
        
        param: x_copy: one-hot-encoded sequences with shape of (1, seq_length, 4)
        param: mutation_size: even number. Total mutation sites
        
        return: list of mutagenized one-hot-encoded sequences with shape of (1*mutation_size*3, seq_length, 4)  

        Example
        --------
        from wrapper import utils
        
        temp_seq = "ACATG"
        temp_seq_one_hot = torch.from_numpy(utils.one_hot(temp_seq))
        temp_seq_one_hot = temp_seq_one_hot.T.unsqueeze(0)
        muts = generate_mutagenesis(temp_seq_one_hot, mutation_size=2)
        print([utils.one_hot_to_sequence(mut) for mut in muts])
        """
        if isinstance(x, torch.Tensor):
            x_copy = copy.deepcopy(x)
            x_copy = x_copy.cpu().numpy()

        _,L,D = x_copy.shape 
        mid = L//2

        if mutation_size>(L-1):
            raise ValueError("Mutation size should be less than half of the sequence length.")
        elif mutation_size%2!=0:
            raise ValueError("Mutation size should be even number.")
        
        x_mut = np.zeros((mutation_size*3, L, D))
        k = 0
        for l in range(mid-int(mutation_size/2), mid+int(mutation_size/2)+1):
            if l==mid:
                continue
            for d in range(3):
                if x_copy[0,l,d]==1:
                    continue
                x_new = copy.deepcopy(x_copy)
                x_new[0,l,:] = 0
                x_new[0,l,d] = 1
                x_mut[k] = x_new
                k += 1
        return x_mut

    def get_score(x: torch.Tensor, model: torch.nn.Module, class_index: int=None, batch_size=32) -> np.ndarray:
        """
        Get score from the model and process based on class_index.

        param: x: torch.Tensor: one-hot-encoded sequences with shape of (batch, seq_length, 4)
        param: model: torch.nn.Module: trained model
        param: class_index: int: class index to choose from the model output 0 (control) or 1 (case)

        return: score: np.ndarray: model predictions with shape of (batch, 1)

        Example
        --------
        get_score(seq_fasta_one_hot[0:30,:,:], model, class_index=0)
        """
        if class_index not in [0,1]:
            raise ValueError("class_index should be either 0 or 1.")

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cuda_device}")
            if not x.is_cuda:
                x = x.cuda(cuda_device)
            if next(model.parameters()).device == torch.device("cpu"): 
                model = model.cuda(cuda_device)
        else:
            device = torch.device("cpu")
        
        if x.shape[2]==4:
            x = x.permute(0,2,1)
            
        x_loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)
        score = torch.Tensor().to(device)

        model.eval()

        for data in x_loader:
            pred = model(data) # [batch_size, 2]
            score = torch.cat([score, pred.detach()])
        score = score.detach().cpu().numpy()
        if class_index == None:
            # Square root of sum of squares of all classes for each sequence
            score = np.sqrt(np.sum(score**2, axis=-1, keepdims=True)) # [batch_size, 1]
        else:
            # Choosing class based on class_index
            score = score[:,class_index]
        return score

    with torch.no_grad():
        # generate mutagenized sequences
        if verbose:
            print("Generating mutagenized sequences...")
        x_mut = generate_mutagenesis(x, mutation_size)
        x_mut = torch.from_numpy(x_mut).float().to(device) # (mutation_size*3, seq_length, 4)  
        
        # get baseline wildtype score
        if verbose:
            print("Getting baseline wildtype score...")
        wt_score = get_score(x, model, class_index, batch_size) # [1,] 
        predictions = get_score(x_mut, model, class_index, batch_size) # [mutation_size * 3,]
        delta = predictions - wt_score 
        # mutation_size = 4
        # temp1 = torch.Tensor([[1,2,3,4,5,6,7,8,9,10,11,12]])
        # temp1 = temp1.reshape(mutation_size,3).T
        # print(temp1)
        delta = delta.reshape(mutation_size,3).T # [3, mutation_size]

    return delta


def plot_mutagenesis(mutation_data: np.ndarray, mutation_size: int = 50, title: str = "Mutagenesis") -> None:
    """

    """

    data = mutation_data
    mutation_size = 500
    x = np.arange(-mutation_size/2, mutation_size/2)
    # Function to process data chunks

    def process_chunk(slice_range):
        start, end = slice_range
        x_batch = np.tile(x, (end - start, 1)).ravel()
        y_batch = data[start:end].ravel()
        colors = np.where(y_batch > 0, 'green', 'red')
        return (x_batch, y_batch, colors)

    # Split data indices into chunks for multiprocessing
    def create_chunks(data, num_chunks):
        chunk_size = len(data) // num_chunks
        return [(i, min(i + chunk_size, len(data))) for i in range(0, len(data), chunk_size)]

    num_processes = 10  # or the number of cores you have
    chunks = create_chunks(data, num_processes)

    # Use multiprocessing to process data
    with Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    # Plotting
    fig, ax = plt.subplots()
    for x_batch, y_batch, colors in results:
        ax.scatter(x_batch, y_batch, color=colors, alpha=0.3, s=3)

    # Setting labels and titles
    ax.set_xlabel('Position (centered at m6A site)')
    ax.set_ylabel('$\Delta$ m6A Level')
    ax.set_title('Control Delta Mutagenesis')
    plt.show()


if __name__ == "__main__":
    import numpy as np 
    import torch 
    import sys
    sys.path.append("/binf-isilon/renniegrp/vpx267/ucph_thesis/")
    from model import ConfigurableModel
    from wrapper import utils
    from tqdm import tqdm
    from wrapper import utils
    import copy

    for fold in [3,4,5]:
        print(f"Starting generating mutagenesis for fold {fold}")

        print("Loading model...")
        config = {'lr': 0.001, 'weight_decay': 0.1, 'cnn_first_filter': 24, 'cnn_first_kernel_size': 7, 'cnn_length': 3, 'cnn_filter': 32, 'cnn_kernel_size': 7, 'bilstm_layer': 3, 'bilstm_hidden_size': 256, 'fc_size': 256}           
        model = ConfigurableModel(input_channel=4, cnn_first_filter=config["cnn_first_filter"], cnn_first_kernel_size=config["cnn_first_kernel_size"],
                            cnn_other_filter=config["cnn_filter"], cnn_other_kernel_size=config["cnn_kernel_size"], bilstm_layer=config["bilstm_layer"], bilstm_hidden_size=config["bilstm_hidden_size"], fc_size=config["fc_size"],
                            output_size=2)
        model_weight = torch.load(f"/binf-isilon/renniegrp/vpx267/ucph_thesis/data/outputs/models/trained_model_{fold}th_fold_dual_outputs_m6A_info-no_promoter-False_fixed_tune.pkl",
                                map_location=torch.device('cpu'))
        model.load_state_dict(model_weight)
        model.eval()

        seq_fasta_test_path = f"/binf-isilon/renniegrp/vpx267/ucph_thesis/data/dual_outputs/motif_fasta_test_SPLIT_{fold}.fasta"
        seq_fasta_one_hot = utils.create_seq_tensor(seq_fasta_test_path)

        seqs_number = seq_fasta_one_hot.shape[0]
        input_seq = seq_fasta_one_hot[0:seqs_number,:,:]
        mutation_size=500

        print(f"Input test shape: {input_seq.shape} | Mutation size = 500")
        ctrl_mutant_score_diff = []
        case_mutant_score_diff = []
        for i in tqdm(range (input_seq.shape[0]), desc=f"Mutagenesis Score fold {fold}"):
            ctrl_mutant_score_diff.append(mutagenesis(x=input_seq[i:i+1,:,:].transpose(1,2), model=model, mutation_size=mutation_size, class_index=0, verbose=False))
            case_mutant_score_diff.append(mutagenesis(x=input_seq[i:i+1,:,:].transpose(1,2), model=model, mutation_size=mutation_size, class_index=1, verbose=False))
            
        ctrl_mutant_score_diff = np.vstack(np.array(ctrl_mutant_score_diff))
        case_mutant_score_diff = np.vstack(np.array(case_mutant_score_diff))

        print("Saving...")
        np.savez(f"/binf-isilon/renniegrp/vpx267/ucph_thesis/analysis/mutagenesis_score_fold_{fold}_fixed_tune.npz", ctrl_mutant_score_diff=ctrl_mutant_score_diff, case_mutant_score_diff=case_mutant_score_diff)
