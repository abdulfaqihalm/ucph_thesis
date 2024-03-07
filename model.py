from torch import nn
import numpy as np

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
                                nn.Sigmoid()
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
    def __init__(self, input_size=None) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0), #[bs, 8, 48]
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),#[bs 32 48]
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 32 24]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 24]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0) #[bs 128 12]
            )
        self.biiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        in_features_1 = (input_size - 7) // 2 + 1   # 48
        in_features_2 = (in_features_1 - 2) // 2 + 1    # 24
        in_features_3 = (in_features_2 - 2) // 2 + 1    # 12
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.FC1= nn.Sequential(nn.Linear(in_features=2*128*in_features_3,out_features=1024),
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
                                nn.Sigmoid()
                                )


    # Define the model    
    def forward(self, x) -> None:
        out = self.NaiveCNN(x)
        out = out.permute(0, 2, 1) #[bs 12 128]
        out, h = self.biiLSTM(out)
        out = self.Flatten(out) # flatten output
        out = self.FC1(out)
        out = self.FC2(out)
        return out