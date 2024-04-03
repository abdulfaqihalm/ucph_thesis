import numpy as np 
import torch 
import torch.nn as nn
from tqdm import tqdm 

#TODO: Implement functions for mutating sequence
def mutagenesis(x: torch.Tensor, model: torch.nn.Module, mutation_size :int=150, class_index: None|int=None, verbose: bool=True) -> np.ndarray:
    """ 
    in silico mutagenesis. modified version from https://github.com/p-koo/tfomics/blob/master/tfomics/explain.py

    input: x: one-hot-encoded sequences with shape of (batch, seq_length, 4)
    input: model: trained model 

    """
    device = (torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu"))

    if device == torch.device("cuda"):
        if not x.is_cuda:
            x = x.cuda()
        if next(model.parameters()).device == torch.device("cpu"): 
            model = model.cuda()
    
    model.eval()

    def generate_mutagenesis(x: torch.Tensor) -> np.ndarray:
        """
        Create a single point mutation at each position of a SINGLE input sequence. 
        
        param: x: one-hot-encoded sequences with shape of (1, seq_length, 4)
        return: list of mutagenized one-hot-encoded sequences with shape of (batch*seq_length*4, seq_length, 4)  
        """
        _,L,A = x.shape 
        x_mut = []
        for l in range(L):
            for a in range(A):
                x_new = np.copy(x.cpu()) # move to CPU if x in GPU
                x_new[0,l,:] = 0
                x_new[0,l,a] = 1
                x_mut.append(x_new)
        return np.concatenate(x_mut, axis=0)

    def reconstruct_map(x: torch.Tensor, predictions: np.ndarray) -> np.ndarray:
        """
        Reconstruct the mutagenezied score

        param: x: one-hot-encoded sequences with shape of (batch, seq_length, 4)
        predictions: model predictions with shape of (batch*seq_length*4)

        return: reconstructed mut_score with shape of (1, seq_length, 4)
        """
        _,L,A = x.shape 
        
        mut_score = np.zeros((1,L,A))
        k = 0
        for l in range(L):
            for a in range(A):
                mut_score[0,l,a] = predictions[k]
                k += 1
        return mut_score

    def get_score(x: torch.Tensor, model: torch.nn.Module, class_index: int=None, batch_size=256) -> np.ndarray:
      """
      Get score from the model and process based on class_index.
      """
      x_loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)
      score = torch.Tensor().to("cpu")

      # Batching to prevent the GPU from running out of memory
      epoch_iterator = tqdm(x_loader, desc="Data Loader Iteration")
      for _, data in enumerate(epoch_iterator):
          pred = model.predict(data)
          torch.cat([score, pred.cpu().detach()])
      score = score.numpy()
      if class_index == None:
          # Square root of sum of squares of all classes
          score = np.sqrt(np.sum(score**2, axis=-1, keepdims=True)) 
      else:
          # Choosing class based on class_index
          score = score[:,class_index]
      return score.cpu().numpy()

    with torch.no_grad():
        # generate mutagenized sequences
        if verbose:
            print("Generating mutagenized sequences...")
        x_mut = generate_mutagenesis(x)
        
        # get baseline wildtype score
        if verbose:
            print("Getting baseline wildtype score...")
        wt_score = get_score(x, model, class_index) # [batch_size,1] 
        predictions = get_score(x_mut, model, class_index) # [batch_size * 4, 1]

        # reshape mutagenesis predictiosn
        if verbose:
            print("Reconstructing mutagenized score...")
        mut_score = reconstruct_map(predictions) # (1, seq_length, 4)

    # Back to cpu
    if device == torch.device("cuda"):
      x = x.cpu()
      model = model.cpu()

    return mut_score - wt_score # Non-mutagenized difference would be zero 