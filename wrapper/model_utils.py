import torch
from torch import nn 
import torch.nn.functional as F
import numpy as np 
import pickle

class BahdanauAttentionV2(nn.Module):
    """
    Input: 
    """
    def __init__(self, in_features, hidden_size) -> None:
        super().__init__()
        self.Wa = nn.Linear(in_features=in_features, out_features=hidden_size) # (batch_size, seq_len,hidden_size)
        self.Ua = nn.Linear(in_features=in_features, out_features=hidden_size) # (num_directions*num_layers, batch_size, RNN_hidden_size)
        self.Va = nn.Linear(in_features=hidden_size, out_features=1)
    
    def forward(self, query, keys):
        """
        param:  query:  previous hidden state of decoder
        param:  keys:   last layer encoder outputs 


        RNN_out: (batch_size, seq_len, num_directions*RNN_hidden_size)
        RNN_hn: (num_directions*num_layers, batch_size, RNN_hidden_size)
        """
        weights = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))   #self.Attention(h_n,output)
        weights = F.softmax(weights, dim=1)

        return

class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """
    def __init__(self,in_features, hidden_units,num_task):
        super(BahdanauAttention,self).__init__()
        self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

        score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
        #print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values,attention_weights)
        context_vector = torch.transpose(context_vector,1,2)
        return context_vector, attention_weights

class EmbeddingSeq(nn.Module):
    def __init__(self,weight_dict_path):
        """
        Inputs:
            weight_dict_path: path of pre-trained embeddings of RNA/dictionary
        """
        super(EmbeddingSeq,self).__init__()
        weight_dict = pickle.load(open(weight_dict_path,'rb'))

        weights = torch.FloatTensor(list(weight_dict.values())).cuda()
        num_embeddings = len(list(weight_dict.keys()))
        embedding_dim = 300

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim)
        self.embedding.weight = nn.Parameter(weights)
        self.embedding.weight.requires_grad = False

    def forward(self,x):

        out = self.embedding(x.type(torch.cuda.LongTensor))

        return out

if __name__ == "__main__":
    embd = EmbeddingSeq('/binf-isilon/renniegrp/vpx267/ucph_thesis/data/embeddings/MultiRM/Embeddings/embeddings_12RM.pkl')
    coba = embd("GGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGG")
    print(coba)