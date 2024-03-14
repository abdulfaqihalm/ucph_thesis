from torch import nn
import torch
import numpy as np
from wrapper.model_utils import BahdanauAttention, EmbeddingHmm

class NaiveModelV1(nn.Module):
    ## Naive model from MultiRM Paper by Song et al.
    def __init__(self, input_size=None) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0), #[bs, 8, 48]
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),#[bs 32 48]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 32 24]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 24]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0) #[bs 128 12]
            )
        in_features_1 = (input_size - 7) // 2 + 1   # 48
        in_features_2 = (in_features_1 - 2) // 2 + 1    # 24
        in_features_3 = (in_features_2 - 2) // 2 + 1    # 12
        self.Flatten = nn.Flatten() # 128*12
        self.FC1= nn.Sequential(nn.Linear(in_features=128*in_features_3,out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )
        self.FC2 = nn.Sequential(nn.Linear(in_features=1024,out_features=256),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=256,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=64,out_features=1),
                                nn.Hardtanh(min_val=0, max_val=1)
                                )


    # Define the model    
    def forward(self, x) -> None:
        out = self.NaiveCNN(x)
        out = self.Flatten(out) # flatten output
        out = self.FC1(out)
        out = self.FC2(out)
        return out


class NaiveModelV2(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_channel=None, cnn_first_filter=8) -> None:
        super().__init__()
        # HARD CODED 1001 
        self.input_size=1001
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=input_channel,out_channels=cnn_first_filter,kernel_size=7,stride=2,padding=0), #[bs, 8, 498]
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),#[bs 32 498]
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 32 249]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 249]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
             #nn.MaxPool1d(kernel_size=2,padding=0)
            )
        self.biiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True, num_layers=3)
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.FC1= nn.Sequential(nn.Linear(in_features=2*128*249,out_features=1024),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(in_features=1024,out_features=512),
                                nn.ReLU(),
                                nn.Dropout(p=0.2)
                                )
        self.FC2 = nn.Sequential(nn.Linear(in_features=512,out_features=256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(in_features=256,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(in_features=64,out_features=1),
                                nn.Hardtanh(0, 1)
                                )


    # Define the model    
    def forward(self, x) -> None:
        out = self.NaiveCNN(x)
        out = out.permute(0, 2, 1) #[bs 12 128]
        out, h = self.biiLSTM(out)
        out = self.Flatten(out) # flatten output
        out = self.FC1(out)
        out = self.FC2(out)
        out = out*100
        return out



class NaiveModelV3(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_size=None) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=8,kernel_size=11,stride=2,padding=0), #[bs, 8, 496]
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8,out_channels=64,kernel_size=3,stride=1,padding=1),#[bs 64 496]
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 64 248]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 248]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0), #[bs 128 124]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),#[bs 128 124]
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0) #[bs 256 62]
            )
        self.biiLSTM = nn.LSTM(input_size=256,hidden_size=256,batch_first=True,bidirectional=True, num_layers=2)
        cnn_features = 62
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.FC1= nn.Sequential(nn.Linear(in_features=2*256*cnn_features,out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )
        self.FC2 = nn.Sequential(nn.Linear(in_features=1024,out_features=256),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=256,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=64,out_features=1),
                                nn.Hardtanh(0, 1)
                                )
    # Define the model    
    def forward(self, x) -> None:
        out = self.NaiveCNN(x)
        out = out.permute(0, 2, 1) #[bs 12 128]
        out, h = self.biiLSTM(out)
        out = self.Flatten(out) # flatten output
        out = self.FC1(out)
        out = self.FC2(out)
        out = out*100
        return out


class NaiveModelV4(nn.Module):
    ## Naive model from MultiRM Paper by Song et al.
    def __init__(self, input_size=None) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0)
                        )
        self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        in_features_1 = (input_size - 7) // 2 + 1   # 48
        in_features_2 = (in_features_1 - 2) // 2 + 1    # 24
        in_features_3 = (in_features_2 - 2) // 2 + 1    # 12
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.Attention = BahdanauAttention(in_features=256,hidden_units=10,num_task=1)
        self.FC1 = nn.Sequential(nn.Linear(in_features=256,out_features=128),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=128,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=64,out_features=1),
                                nn.Hardtanh(0, 1)
                                )
    # Define the model    
    def forward(self, x) -> None:
        x = self.NaiveCNN(x)
        batch_size, features, seq_len = x.size()
        x = x.view(batch_size,seq_len, features) # parepare input for LSTM
        output, (h_n, c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1]) # pareprae input for Attention
        context_vector,attention_weights = self.Attention(h_n,output) # Attention (batch_size, num_task, unit)num_task, unit)
        out = self.Flatten(context_vector[:,0,:]) # flatten output
        out = self.FC1(out)
        out = out*100
        return out
    

class MultiRMModel(nn.Module):

    def __init__(self,num_task):
        super().__init__()

        # self.num_task = num_task
        # self.use_embedding = use_embedding
        # if self.use_embedding:
        #     self.embed = EmbeddingHmm(t=3,out_dims=256) # hmm
        #     self.NaiveBiLSTM = nn.LSTM(input_size=256,hidden_size=256,batch_first=True,bidirectional=True)
        # else:
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0),
                        nn.Dropout(p=0.2)
                        )

        self.NaiveBiLSTM = nn.LSTM(input_size=4,hidden_size=256,batch_first=True,bidirectional=True)

        self.Attention = BahdanauAttention(in_features=512,hidden_units=100,num_task=num_task)
        self.NaiveFC1 = nn.Sequential(
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        batch_size = x.size()[0]
        # x = torch.transpose(x,1,2)

        output,(h_n,c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1])
        context_vector,attention_weights = self.Attention(h_n,output)
        # print(attention_weights.shape)
        out = self.NaiveFC1(context_vector[:,0,:])
        out = torch.squeeze(out, dim=-1)
        return out.unsqueeze(1)



if __name__=="__main__":
    #print(NaiveModelV2(1001))
    kernel_size=7
    stride=2 
    padding=0
    dilation=1
    lout = ((1001 + (2*padding) - (dilation * (kernel_size - 1 )) - 1)/stride) + 1
    print(lout)
    kernel_size=3 
    stride=1
    padding=1
    lout = ((lout + (2*padding) - (dilation * (kernel_size - 1 )) - 1)/stride) + 1
    print(lout)
    lout = (lout + 2*0 - 1*(2-1)-1)/2 + 1
    # print(lout)
    # kernel_size=3 
    # stride=1
    # padding=1
    # lout = ((lout + (2*padding) - (dilation * (kernel_size - 1 )) - 1)/stride) + 1
    # print(lout)
    # lout = (lout + 2*0 - 1*(2-1)-1)/2 + 1
    # print(lout)